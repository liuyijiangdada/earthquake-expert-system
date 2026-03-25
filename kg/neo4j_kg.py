#!/usr/bin/env python3
# 基于 Neo4j 的地震知识图谱（真实事件目录 + 应急知识关系）

import json
import os
import sys
from typing import Optional

import pandas as pd
from neo4j import GraphDatabase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

# 用于从文本中归并省级 Region（与 app 中地区列表一致）
_REGION_NAMES = [
    "四川", "云南", "青海", "西藏", "新疆", "甘肃", "河北", "台湾", "广东", "辽宁",
    "北京", "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北",
    "湖南", "广西", "海南", "重庆", "贵州", "陕西", "吉林", "黑龙江", "内蒙古",
    "宁夏", "香港", "澳门",
]


def _node_to_dict(node) -> dict:
    d = dict(node)
    return {
        "id": d.get("id"),
        "name": d.get("name"),
        "time": d.get("time"),
        "magnitude": float(d.get("magnitude") or 0),
        "depth": float(d.get("depth") or 0),
        "location": d.get("location"),
        "latitude": float(d.get("latitude") or 0),
        "longitude": float(d.get("longitude") or 0),
        "intensity": d.get("intensity"),
        "description": d.get("description"),
    }


def _row_to_eq(record) -> dict:
    return _node_to_dict(record["e"])


def _infer_region_from_location(location: str) -> Optional[str]:
    if not location:
        return None
    for r in _REGION_NAMES:
        if r in location:
            return r
    return None


class Neo4jKG:
    """地震事件 + 区域节点 + 应急知识主题与步骤关系。"""

    def __init__(self):
        self.config = Config()
        self._driver = None

    def _connect(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.NEO4J_URI,
                auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD),
            )
        return self._driver

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _ensure_schema(self, session):
        session.run(
            """
            CREATE CONSTRAINT earthquake_id_unique IF NOT EXISTS
            FOR (e:Earthquake) REQUIRE e.id IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT region_name_unique IF NOT EXISTS
            FOR (r:Region) REQUIRE r.name IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT emergency_topic_id_unique IF NOT EXISTS
            FOR (t:EmergencyTopic) REQUIRE t.id IS UNIQUE
            """
        )
        session.run(
            """
            CREATE CONSTRAINT guidance_step_id_unique IF NOT EXISTS
            FOR (s:GuidanceStep) REQUIRE s.id IS UNIQUE
            """
        )

    def _import_emergency_knowledge(self, session, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        topics = data.get("topics", [])
        print(f"导入应急知识主题 {len(topics)} 个…")
        for topic in topics:
            tid = topic["id"]
            session.run(
                """
                MERGE (t:EmergencyTopic {id: $id})
                SET t.title = $title, t.category = $category, t.source = $source
                """,
                id=tid,
                title=topic.get("title", ""),
                category=topic.get("category", ""),
                source=topic.get("source", ""),
            )
            for step in topic.get("steps", []):
                oid = int(step["order"])
                sid = f"{tid}_step_{oid}"
                session.run(
                    """
                    MATCH (t:EmergencyTopic {id: $tid})
                    MERGE (s:GuidanceStep {id: $sid})
                    SET s.text = $text, s.order = $ord
                    MERGE (t)-[hs:HAS_STEP {order: $ord}]->(s)
                    """,
                    tid=tid,
                    sid=sid,
                    text=step.get("text", ""),
                    ord=oid,
                )
        for rel in data.get("topic_relations", []):
            session.run(
                """
                MATCH (a:EmergencyTopic {id: $fid}), (b:EmergencyTopic {id: $tid})
                MERGE (a)-[r:RELATES_TO]->(b)
                SET r.relation_type = $rtype
                """,
                fid=rel["from_id"],
                tid=rel["to_id"],
                rtype=rel.get("relation_type", ""),
            )
        print("应急知识关系导入完成。")

    def _import_real_earthquake_catalog(self, session, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("earthquakes", [])
        print(f"导入真实地震目录 {len(rows)} 条…")
        for row in rows:
            eid = str(row["id"])
            region = row.get("region") or _infer_region_from_location(row.get("location", ""))
            session.run(
                """
                MERGE (e:Earthquake {id: $id})
                SET e.name = $name,
                    e.time = $time,
                    e.magnitude = toFloat($magnitude),
                    e.depth = toFloat($depth),
                    e.location = $location,
                    e.latitude = toFloat($latitude),
                    e.longitude = toFloat($longitude),
                    e.intensity = $intensity,
                    e.description = $description
                """,
                id=eid,
                name=str(row.get("name", "")),
                time=str(row.get("time", "")),
                magnitude=float(row.get("magnitude", 0)),
                depth=float(row.get("depth", 0)),
                location=str(row.get("location", "")),
                latitude=float(row.get("latitude", 0)),
                longitude=float(row.get("longitude", 0)),
                intensity=str(row.get("intensity", "")),
                description=str(row.get("description", "")),
            )
            if region:
                session.run(
                    """
                    MATCH (e:Earthquake {id: $eid})
                    MERGE (r:Region {name: $rname})
                    MERGE (e)-[:OCCURRED_IN]->(r)
                    """,
                    eid=eid,
                    rname=region,
                )
            for tid in row.get("links_to_topics", []) or []:
                session.run(
                    """
                    MATCH (e:Earthquake {id: $eid}), (t:EmergencyTopic {id: $tid})
                    MERGE (e)-[:SUGGESTS_TOPIC]->(t)
                    """,
                    eid=eid,
                    tid=tid,
                )
        print("真实地震目录与关联导入完成。")

    def _import_csv(self, session):
        path = self.config.EARTHQUAKE_DATA_FILE
        if not os.path.isfile(path):
            print(f"警告: 数据文件不存在 {path}，跳过 CSV 导入。")
            return
        df = pd.read_csv(path)
        print(f"向 Neo4j 导入 {len(df)} 条地震数据（CSV，兼容旧版）...")
        for _, row in df.iterrows():
            eid = str(row["id"])
            loc = str(row.get("location", ""))
            region = _infer_region_from_location(loc)
            session.run(
                """
                MERGE (e:Earthquake {id: $id})
                SET e.name = $name,
                    e.time = $time,
                    e.magnitude = toFloat($magnitude),
                    e.depth = toFloat($depth),
                    e.location = $location,
                    e.latitude = toFloat($latitude),
                    e.longitude = toFloat($longitude),
                    e.intensity = $intensity,
                    e.description = $description
                """,
                id=eid,
                name=str(row["name"]),
                time=str(row["time"]),
                magnitude=float(row["magnitude"]),
                depth=float(row["depth"]),
                location=loc,
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                intensity=str(row["intensity"]),
                description=str(row["description"]),
            )
            if region:
                session.run(
                    """
                    MATCH (e:Earthquake {id: $eid})
                    MERGE (r:Region {name: $rname})
                    MERGE (e)-[:OCCURRED_IN]->(r)
                    """,
                    eid=eid,
                    rname=region,
                )
        print("Neo4j CSV 导入完成。")

    def run(self):
        """空库时优先加载「真实目录 + 应急知识」JSON；否则回退仅 CSV。"""
        driver = self._connect()
        try:
            with driver.session() as session:
                self._ensure_schema(session)
                n = session.run(
                    "MATCH (e:Earthquake) RETURN count(e) AS c"
                ).single()["c"]
                if n == 0:
                    print("初始化 Neo4j 知识图谱...")
                    cat = self.config.REAL_EARTHQUAKE_CATALOG_FILE
                    emg = self.config.EMERGENCY_KNOWLEDGE_FILE
                    if os.path.isfile(cat) and os.path.isfile(emg):
                        self._import_emergency_knowledge(session, emg)
                        self._import_real_earthquake_catalog(session, cat)
                    elif os.path.isfile(self.config.EARTHQUAKE_DATA_FILE):
                        self._import_csv(session)
                    else:
                        print("警告: 未找到 real_earthquakes_catalog.json / emergency_knowledge.json 或 earthquake_data.csv")
                print("Neo4j 知识图谱就绪。")
        except Exception as e:
            print(f"Neo4j 连接或初始化失败: {e}")
            raise

    @property
    def earthquakes(self):
        return self.query_all_earthquakes()

    def count_earthquakes(self):
        driver = self._connect()
        with driver.session() as session:
            return session.run(
                "MATCH (e:Earthquake) RETURN count(e) AS c"
            ).single()["c"]

    def query_all_earthquakes(self):
        driver = self._connect()
        with driver.session() as session:
            result = session.run(
                "MATCH (e:Earthquake) RETURN e ORDER BY e.time DESC"
            )
            return [_row_to_eq(r) for r in result]

    def query_earthquakes_by_region(self, region):
        driver = self._connect()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Earthquake)
                WHERE e.location CONTAINS $region
                RETURN e ORDER BY e.time DESC
                """,
                region=region,
            )
            return [_row_to_eq(r) for r in result]

    def query_earthquakes_by_magnitude(self, min_magnitude, max_magnitude=10.0):
        driver = self._connect()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Earthquake)
                WHERE toFloat(e.magnitude) >= $min_m AND toFloat(e.magnitude) <= $max_m
                RETURN e ORDER BY e.magnitude DESC
                """,
                min_m=float(min_magnitude),
                max_m=float(max_magnitude),
            )
            return [_row_to_eq(r) for r in result]

    def query_earthquakes_by_depth(self, min_depth, max_depth):
        driver = self._connect()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Earthquake)
                WHERE toFloat(e.depth) >= $min_d AND toFloat(e.depth) <= $max_d
                RETURN e ORDER BY e.depth
                """,
                min_d=float(min_depth),
                max_d=float(max_depth),
            )
            return [_row_to_eq(r) for r in result]

    def query_earthquakes_by_time_range(self, start_time, end_time):
        driver = self._connect()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Earthquake)
                WHERE e.time >= $start_t AND e.time <= $end_t
                RETURN e ORDER BY e.time
                """,
                start_t=start_time,
                end_t=end_time,
            )
            return [_row_to_eq(r) for r in result]

    def query_related_earthquakes(self, earthquake_id):
        driver = self._connect()
        with driver.session() as session:
            loc = session.run(
                "MATCH (e:Earthquake {id: $id}) RETURN e.location AS loc",
                id=earthquake_id,
            ).single()
            if not loc or loc["loc"] is None:
                return []
            result = session.run(
                """
                MATCH (e:Earthquake)
                WHERE e.location = $loc AND e.id <> $id
                RETURN e
                """,
                loc=loc["loc"],
                id=earthquake_id,
            )
            return [_row_to_eq(r) for r in result]

    def query_earthquake_details(self, earthquake_id):
        eq = None
        driver = self._connect()
        with driver.session() as session:
            rec = session.run(
                "MATCH (e:Earthquake {id: $id}) RETURN e",
                id=earthquake_id,
            ).single()
            if rec:
                eq = _node_to_dict(rec["e"])
        if not eq:
            return None
        mag = float(eq["magnitude"])
        depth = float(eq["depth"])
        level = Neo4jKG._magnitude_level(mag)
        dcat = Neo4jKG._depth_category(depth)
        return {
            "e": eq,
            "rgn": {"id": f"region_{eq['location']}", "name": eq["location"]},
            "m": {"id": f"mag_{mag}", "value": mag, "level": level},
            "d": {"id": f"depth_{depth}", "value": depth, "unit": "km", "category": dcat},
            "t": {"id": f"time_{eq['time']}", "timestamp": eq["time"]},
        }

    def query_emergency_context(self, user_text: str) -> str:
        """当用户询问避险、怎么办等时，从图谱抽取应急要点供模型参考。"""
        triggers = (
            "怎么办", "如何做", "怎样", "避险", "避震", "应急", "自救", "互救",
            "余震", "室内", "室外", "高楼", "学校", "准备", "演练", "逃生",
        )
        if not any(t in user_text for t in triggers):
            return ""
        driver = self._connect()
        lines = []
        with driver.session() as session:
            result = session.run(
                """
                MATCH (t:EmergencyTopic)
                OPTIONAL MATCH (t)-[hs:HAS_STEP]->(s:GuidanceStep)
                RETURN t.id AS tid, t.title AS title, t.category AS cat, t.source AS src,
                       hs.order AS ord, s.text AS stext
                ORDER BY t.id, hs.order
                """
            )
            by_topic = {}
            for r in result:
                tid = r["tid"]
                if tid not in by_topic:
                    by_topic[tid] = {
                        "title": r["title"],
                        "cat": r["cat"],
                        "src": r["src"],
                        "steps": [],
                    }
                if r["stext"]:
                    by_topic[tid]["steps"].append((r["ord"], r["stext"]))
        for _, info in by_topic.items():
            lines.append(f"《{info['title']}》（{info['cat']}）")
            for ord_, txt in sorted(info["steps"], key=lambda x: x[0] or 0):
                lines.append(f"  {int(ord_)}. {txt}")
            if info.get("src"):
                lines.append(f"  参考说明：{info['src']}")
            lines.append("")
        if not lines:
            return ""
        return "【知识图谱·应急避险要点】\n" + "\n".join(lines).strip() + "\n\n"

    @staticmethod
    def _magnitude_level(magnitude):
        if magnitude < 3.0:
            return "微震"
        if magnitude < 4.0:
            return "小震"
        if magnitude < 5.0:
            return "中震"
        if magnitude < 6.0:
            return "强震"
        if magnitude < 7.0:
            return "大地震"
        if magnitude < 8.0:
            return "巨大地震"
        return "特大地震"

    @staticmethod
    def _depth_category(depth):
        if depth < 70:
            return "浅源地震"
        if depth < 300:
            return "中源地震"
        return "深源地震"

    def upsert_earthquake(self, eq: dict):
        """单条写入或更新（用于实时数据），并尽量关联 Region。"""
        driver = self._connect()
        eid = str(eq["id"])
        loc = str(eq.get("location", ""))
        region = _infer_region_from_location(loc)
        with driver.session() as session:
            session.run(
                """
                MERGE (e:Earthquake {id: $id})
                SET e.name = $name,
                    e.time = $time,
                    e.magnitude = toFloat($magnitude),
                    e.depth = toFloat($depth),
                    e.location = $location,
                    e.latitude = toFloat($latitude),
                    e.longitude = toFloat($longitude),
                    e.intensity = $intensity,
                    e.description = $description
                """,
                id=eid,
                name=str(eq.get("name", "")),
                time=str(eq.get("time", "")),
                magnitude=float(eq.get("magnitude", 0)),
                depth=float(eq.get("depth", 0)),
                location=loc,
                latitude=float(eq.get("latitude", 0)),
                longitude=float(eq.get("longitude", 0)),
                intensity=str(eq.get("intensity", "")),
                description=str(eq.get("description", "")),
            )
            if region:
                session.run(
                    """
                    MATCH (e:Earthquake {id: $eid})
                    MERGE (r:Region {name: $rname})
                    MERGE (e)-[:OCCURRED_IN]->(r)
                    """,
                    eid=eid,
                    rname=region,
                )

    def update_from_realtime_data(self):
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.earthquake_api import update_earthquake_data

        try:
            new_earthquakes = update_earthquake_data()
            for eq in new_earthquakes:
                self.upsert_earthquake(eq)
            if new_earthquakes:
                print(f"Neo4j 已合并 {len(new_earthquakes)} 条实时地震数据")
            return len(new_earthquakes)
        except Exception as e:
            print(f"更新实时数据失败: {e}")
            return 0


if __name__ == "__main__":
    kg = Neo4jKG()
    kg.run()
    print("条数:", kg.count_earthquakes())
    print("四川相关:", len(kg.query_earthquakes_by_region("四川")))
    print(kg.query_emergency_context("地震来了室内怎么办")[:200])
