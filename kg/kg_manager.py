#!/usr/bin/env python3
# 知识图谱管理类

import pandas as pd
from py2neo import Graph, Node, Relationship
from config.config import Config
from kg.schema import ENTITY_TYPES, RELATIONSHIP_TYPES, QUERY_TEMPLATES

class KGManager:
    def __init__(self):
        self.config = Config()
        self.graph = Graph(
            self.config.NEO4J_URI,
            auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
        )
    
    def clear_graph(self):
        """清空图数据库"""
        self.graph.run("MATCH (n) DETACH DELETE n")
        print("图数据库已清空")
    
    def import_data(self, data_file):
        """从CSV文件导入数据到知识图谱"""
        df = pd.read_csv(data_file)
        print(f"开始导入 {len(df)} 条数据到知识图谱...")
        
        for _, row in df.iterrows():
            # 创建地震节点
            earthquake = Node(
                "Earthquake",
                id=row['id'],
                name=row['name'],
                time=row['time'],
                magnitude=row['magnitude'],
                depth=row['depth'],
                location=row['location'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                intensity=row['intensity'],
                description=row['description']
            )
            self.graph.create(earthquake)
            
            # 创建区域节点
            region = Node(
                "Region",
                id=f"region_{row['location']}",
                name=row['location']
            )
            # 避免重复创建区域节点
            existing_region = self.graph.nodes.match("Region", id=region['id']).first()
            if not existing_region:
                self.graph.create(region)
            else:
                region = existing_region
            
            # 创建震级节点
            magnitude = Node(
                "Magnitude",
                id=f"mag_{row['magnitude']}",
                value=row['magnitude'],
                level=self._get_magnitude_level(row['magnitude'])
            )
            existing_magnitude = self.graph.nodes.match("Magnitude", id=magnitude['id']).first()
            if not existing_magnitude:
                self.graph.create(magnitude)
            else:
                magnitude = existing_magnitude
            
            # 创建深度节点
            depth = Node(
                "Depth",
                id=f"depth_{row['depth']}",
                value=row['depth'],
                unit="km",
                category=self._get_depth_category(row['depth'])
            )
            existing_depth = self.graph.nodes.match("Depth", id=depth['id']).first()
            if not existing_depth:
                self.graph.create(depth)
            else:
                depth = existing_depth
            
            # 创建时间节点
            time_parts = row['time'].split()
            date_part = time_parts[0]
            time_part = time_parts[1]
            date_components = date_part.split('-')
            time_components = time_part.split(':')
            
            time_node = Node(
                "Time",
                id=f"time_{row['time']}",
                timestamp=row['time'],
                date=date_part,
                time=time_part,
                year=int(date_components[0]),
                month=int(date_components[1]),
                day=int(date_components[2]),
                hour=int(time_components[0]),
                minute=int(time_components[1]),
                second=int(time_components[2])
            )
            existing_time = self.graph.nodes.match("Time", id=time_node['id']).first()
            if not existing_time:
                self.graph.create(time_node)
            else:
                time_node = existing_time
            
            # 创建关系
            # 地震发生在区域
            rel_occurred_in = Relationship(earthquake, "OCCURRED_IN", region)
            self.graph.create(rel_occurred_in)
            
            # 地震有震级
            rel_has_magnitude = Relationship(earthquake, "HAS_MAGNITUDE", magnitude)
            self.graph.create(rel_has_magnitude)
            
            # 地震有深度
            rel_has_depth = Relationship(earthquake, "HAS_DEPTH", depth)
            self.graph.create(rel_has_depth)
            
            # 地震发生在时间
            rel_occurred_at = Relationship(earthquake, "OCCURRED_AT", time_node)
            self.graph.create(rel_occurred_at)
        
        print("数据导入完成！")
    
    def _get_magnitude_level(self, magnitude):
        """根据震级获取震级等级"""
        if magnitude < 3.0:
            return "微震"
        elif magnitude < 4.0:
            return "小震"
        elif magnitude < 5.0:
            return "中震"
        elif magnitude < 6.0:
            return "强震"
        elif magnitude < 7.0:
            return "大地震"
        elif magnitude < 8.0:
            return "巨大地震"
        else:
            return "特大地震"
    
    def _get_depth_category(self, depth):
        """根据深度获取深度类别"""
        if depth < 70:
            return "浅源地震"
        elif depth < 300:
            return "中源地震"
        else:
            return "深源地震"
    
    def query_earthquakes_by_region(self, region):
        """根据区域查询地震"""
        query = QUERY_TEMPLATES["get_earthquakes_by_region"]
        results = self.graph.run(query, region=region).data()
        return [result['e'] for result in results]
    
    def query_earthquakes_by_magnitude(self, min_magnitude):
        """根据震级查询地震"""
        query = QUERY_TEMPLATES["get_earthquakes_by_magnitude"]
        results = self.graph.run(query, min_magnitude=min_magnitude).data()
        return [result['e'] for result in results]
    
    def query_earthquakes_by_time_range(self, start_time, end_time):
        """根据时间范围查询地震"""
        query = QUERY_TEMPLATES["get_earthquakes_by_time_range"]
        results = self.graph.run(query, start_time=start_time, end_time=end_time).data()
        return [result['e'] for result in results]
    
    def query_related_earthquakes(self, earthquake_id):
        """查询相关地震（同一区域的其他地震）"""
        query = QUERY_TEMPLATES["get_related_earthquakes"]
        results = self.graph.run(query, earthquake_id=earthquake_id).data()
        return [result['e2'] for result in results]
    
    def query_earthquake_details(self, earthquake_id):
        """查询地震详细信息"""
        query = QUERY_TEMPLATES["get_earthquake_details"]
        results = self.graph.run(query, earthquake_id=earthquake_id).data()
        return results[0] if results else None
    
    def run(self):
        """运行知识图谱管理流程"""
        print("初始化知识图谱...")
        self.clear_graph()
        self.import_data(self.config.EARTHQUAKE_DATA_FILE)
        print("知识图谱构建完成！")

if __name__ == "__main__":
    kg_manager = KGManager()
    kg_manager.run()