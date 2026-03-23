#!/usr/bin/env python3
# 基于 Neo4j 的地震知识图谱

import os
import sys

import pandas as pd
from neo4j import GraphDatabase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config


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


class Neo4jKG:
    """地震数据存于 Neo4j，节点标签 Earthquake，属性与 CSV 列一致。"""

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

    def _import_csv(self, session):
        path = self.config.EARTHQUAKE_DATA_FILE
        if not os.path.isfile(path):
            print(f"警告: 数据文件不存在 {path}，跳过 Neo4j 初始导入。")
            return
        df = pd.read_csv(path)
        print(f"向 Neo4j 导入 {len(df)} 条地震数据（CSV）...")
        for _, row in df.iterrows():
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
                id=str(row["id"]),
                name=str(row["name"]),
                time=str(row["time"]),
                magnitude=float(row["magnitude"]),
                depth=float(row["depth"]),
                location=str(row["location"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                intensity=str(row["intensity"]),
                description=str(row["description"]),
            )
        print("Neo4j CSV 导入完成。")

    def run(self):
        """连接数据库；若无地震节点则从 CSV 导入。"""
        driver = self._connect()
        try:
            with driver.session() as session:
                self._ensure_schema(session)
                n = session.run(
                    "MATCH (e:Earthquake) RETURN count(e) AS c"
                ).single()["c"]
                if n == 0:
                    print("初始化 Neo4j 知识图谱...")
                    self._import_csv(session)
                print("Neo4j 知识图谱就绪。")
        except Exception as e:
            print(f"Neo4j 连接或初始化失败: {e}")
            raise

    @property
    def earthquakes(self):
        """兼容 InMemoryKG 的 .earthquakes 访问。"""
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
                WHERE e.location = $region
                RETURN e ORDER BY e.time DESC
                """,
                region=region,
            )
            return [_row_to_eq(r) for r in result]

    def query_earthquakes_by_magnitude(self, min_magnitude, max_magnitude=10.0):
        """min <= magnitude <= max；仅传 min 时与旧逻辑一致（上限 10）。"""
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
        """单条写入或更新（用于实时数据）。"""
        driver = self._connect()
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
                id=str(eq["id"]),
                name=str(eq.get("name", "")),
                time=str(eq.get("time", "")),
                magnitude=float(eq.get("magnitude", 0)),
                depth=float(eq.get("depth", 0)),
                location=str(eq.get("location", "")),
                latitude=float(eq.get("latitude", 0)),
                longitude=float(eq.get("longitude", 0)),
                intensity=str(eq.get("intensity", "")),
                description=str(eq.get("description", "")),
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
    print("示例查询(四川):", len(kg.query_earthquakes_by_region("四川")))
