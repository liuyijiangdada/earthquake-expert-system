#!/usr/bin/env python3
# 基于内存的知识图谱实现

import pandas as pd
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class InMemoryKG:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InMemoryKG, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not InMemoryKG._initialized:
            self.config = Config()
            self.earthquakes = []
            self.regions = {}
            self.magnitudes = {}
            self.depths = {}
            self.times = {}
            self.relationships = {}
            self.location_index = {}
            InMemoryKG._initialized = True
    
    def import_data(self, data_file):
        """从CSV文件导入数据到内存知识图谱"""
        df = pd.read_csv(data_file)
        print(f"开始导入 {len(df)} 条数据到内存知识图谱...")
        
        for _, row in df.iterrows():
            # 构建地震数据
            earthquake = {
                'id': row['id'],
                'name': row['name'],
                'time': row['time'],
                'magnitude': row['magnitude'],
                'depth': row['depth'],
                'location': row['location'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'intensity': row['intensity'],
                'description': row['description']
            }
            self.earthquakes.append(earthquake)
            
            # 构建位置索引
            if row['location'] not in self.location_index:
                self.location_index[row['location']] = []
            self.location_index[row['location']].append(earthquake)
            
            # 构建区域数据
            region_id = f"region_{row['location']}"
            if region_id not in self.regions:
                self.regions[region_id] = {
                    'id': region_id,
                    'name': row['location']
                }
            
            # 构建震级数据
            magnitude_id = f"mag_{row['magnitude']}"
            if magnitude_id not in self.magnitudes:
                self.magnitudes[magnitude_id] = {
                    'id': magnitude_id,
                    'value': row['magnitude'],
                    'level': self._get_magnitude_level(row['magnitude'])
                }
            
            # 构建深度数据
            depth_id = f"depth_{row['depth']}"
            if depth_id not in self.depths:
                self.depths[depth_id] = {
                    'id': depth_id,
                    'value': row['depth'],
                    'unit': "km",
                    'category': self._get_depth_category(row['depth'])
                }
            
            # 构建时间数据
            time_id = f"time_{row['time']}"
            time_parts = row['time'].split()
            date_part = time_parts[0]
            time_part = time_parts[1]
            date_components = date_part.split('-')
            time_components = time_part.split(':')
            
            if time_id not in self.times:
                self.times[time_id] = {
                    'id': time_id,
                    'timestamp': row['time'],
                    'date': date_part,
                    'time': time_part,
                    'year': int(date_components[0]),
                    'month': int(date_components[1]),
                    'day': int(date_components[2]),
                    'hour': int(time_components[0]),
                    'minute': int(time_components[1]),
                    'second': int(time_components[2])
                }
            
            # 构建关系
            if row['id'] not in self.relationships:
                self.relationships[row['id']] = {
                    'OCCURRED_IN': region_id,
                    'HAS_MAGNITUDE': magnitude_id,
                    'HAS_DEPTH': depth_id,
                    'OCCURRED_AT': time_id
                }
        
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
    
    def query_all_earthquakes(self):
        """查询所有地震数据"""
        return self.earthquakes
    
    def update_from_realtime_data(self):
        """从实时数据更新知识图谱"""
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from scripts.earthquake_api import update_earthquake_data
            new_earthquakes = update_earthquake_data()
            
            for eq in new_earthquakes:
                self.earthquakes.append(eq)
                # 更新索引
                if eq["location"] not in self.location_index:
                    self.location_index[eq["location"]] = []
                self.location_index[eq["location"]].append(eq)
            
            print(f"知识图谱更新了 {len(new_earthquakes)} 条实时数据")
            return len(new_earthquakes)
        except Exception as e:
            print(f"更新实时数据失败: {e}")
            return 0
    
    def query_earthquakes_by_region(self, region):
        """根据区域查询地震"""
        return [eq for eq in self.earthquakes if eq['location'] == region]
    
    def query_earthquakes_by_magnitude(self, min_magnitude):
        """根据震级查询地震"""
        return [eq for eq in self.earthquakes if eq['magnitude'] >= min_magnitude]
    
    def query_earthquakes_by_time_range(self, start_time, end_time):
        """根据时间范围查询地震"""
        return [eq for eq in self.earthquakes if start_time <= eq['time'] <= end_time]
    
    def query_related_earthquakes(self, earthquake_id):
        """查询相关地震（同一区域的其他地震）"""
        target_eq = next((eq for eq in self.earthquakes if eq['id'] == earthquake_id), None)
        if not target_eq:
            return []
        return [eq for eq in self.earthquakes if eq['location'] == target_eq['location'] and eq['id'] != earthquake_id]
    
    def query_earthquake_details(self, earthquake_id):
        """查询地震详细信息"""
        earthquake = next((eq for eq in self.earthquakes if eq['id'] == earthquake_id), None)
        if not earthquake:
            return None
        
        # 获取关联数据
        rels = self.relationships.get(earthquake_id, {})
        region = self.regions.get(rels.get('OCCURRED_IN'), {})
        magnitude = self.magnitudes.get(rels.get('HAS_MAGNITUDE'), {})
        depth = self.depths.get(rels.get('HAS_DEPTH'), {})
        time_node = self.times.get(rels.get('OCCURRED_AT'), {})
        
        return {
            'e': earthquake,
            'rgn': region,
            'm': magnitude,
            'd': depth,
            't': time_node
        }
    
    def run(self):
        """运行知识图谱管理流程"""
        # 检查是否已经导入数据
        if not self.earthquakes:
            print("初始化内存知识图谱...")
            self.import_data(self.config.EARTHQUAKE_DATA_FILE)
            print("知识图谱构建完成！")
        else:
            print("知识图谱已初始化，跳过数据导入")

if __name__ == "__main__":
    kg = InMemoryKG()
    kg.run()
    
    # 测试查询
    print("\n测试查询功能：")
    print("1. 查询所有地震数据:")
    all_eqs = kg.query_all_earthquakes()
    print(f"  共 {len(all_eqs)} 条地震数据")
    
    print("\n2. 根据区域查询地震（四川宜宾）:")
    results = kg.query_earthquakes_by_region("四川宜宾")
    for eq in results[:5]:
        print(f"  - {eq['name']}: {eq['magnitude']}级")
    
    print("\n3. 根据震级查询地震（>=5.0）:")
    results = kg.query_earthquakes_by_magnitude(5.0)
    for eq in results[:5]:
        print(f"  - {eq['name']}: {eq['magnitude']}级")
    
    print("\n4. 查询地震详细信息（eq_1）:")
    details = kg.query_earthquake_details("eq_1")
    if details:
        print(f"  地震: {details['e']['name']}")
        print(f"  区域: {details['rgn']['name']}")
        print(f"  震级: {details['m']['value']}级 ({details['m']['level']})")
        print(f"  深度: {details['d']['value']}km ({details['d']['category']})")
        print(f"  时间: {details['t']['timestamp']}")
