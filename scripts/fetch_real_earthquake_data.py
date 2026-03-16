#!/usr/bin/env python3
# 获取真实地震数据脚本

import requests
import json
import time
from datetime import datetime, timedelta

def fetch_real_earthquake_data(days=30, limit_per_day=50, min_magnitude=3.0):
    """获取真实地震数据
    
    Args:
        days: 获取过去多少天的数据
        limit_per_day: 每天获取的数据条数
        min_magnitude: 最小震级
    """
    all_earthquakes = []
    
    # 读取现有数据
    try:
        with open("data/earthquake_data.json", "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except:
        existing_data = []
    
    print(f"现有数据：{len(existing_data)} 条")
    
    # 生成日期范围
    for day in range(days):
        start_time = datetime.utcnow() - timedelta(days=day+1)
        end_time = datetime.utcnow() - timedelta(days=day)
        
        print(f"获取 {start_time.strftime('%Y-%m-%d')} 到 {end_time.strftime('%Y-%m-%d')} 的数据...")
        
        # 构建API请求
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": start_time.isoformat(),
            "endtime": end_time.isoformat(),
            "limit": limit_per_day,
            "minmagnitude": min_magnitude,
            "orderby": "time-asc"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # 处理数据
            for feature in data.get("features", []):
                properties = feature.get("properties", {})
                geometry = feature.get("geometry", {})
                coordinates = geometry.get("coordinates", [0, 0, 0])
                
                # 转换为中国时区
                time_str = datetime.fromtimestamp(properties.get("time")/1000).strftime("%Y-%m-%d %H:%M:%S")
                
                earthquake = {
                    "id": f"eq_{int(time.time())}_{len(all_earthquakes)}",
                    "name": properties.get("place", "未知地区"),
                    "location": properties.get("place", "未知地区"),
                    "time": time_str,
                    "magnitude": float(properties.get("mag", 0)),
                    "depth": int(abs(coordinates[2])),
                    "latitude": float(coordinates[1]),
                    "longitude": float(coordinates[0]),
                    "intensity": "未知",
                    "description": f"震级{properties.get('mag', 0)}级地震"
                }
                
                all_earthquakes.append(earthquake)
            
            print(f"  成功获取 {len(data.get('features', []))} 条数据")
            
            # 避免API限流
            time.sleep(2)
            
        except Exception as e:
            print(f"  获取数据失败: {e}")
            # 继续尝试下一天
            time.sleep(5)
            continue
    
    # 合并数据
    updated_data = existing_data + all_earthquakes
    
    # 去重（根据位置和时间）
    unique_earthquakes = []
    seen = set()
    
    for eq in updated_data:
        key = (eq["location"], eq["time"])
        if key not in seen:
            seen.add(key)
            unique_earthquakes.append(eq)
    
    # 限制数据数量为1000条
    if len(unique_earthquakes) > 1000:
        unique_earthquakes = unique_earthquakes[:1000]
    
    # 保存数据
    with open("data/earthquake_data.json", "w", encoding="utf-8") as f:
        json.dump(unique_earthquakes, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据获取完成！")
    print(f"新增数据：{len(all_earthquakes)} 条")
    print(f"去重后数据：{len(unique_earthquakes)} 条")
    print(f"总数据量：{len(unique_earthquakes)} 条")
    
    return unique_earthquakes

if __name__ == "__main__":
    print("开始获取真实地震数据...")
    print("使用USGS地震数据API")
    print("获取过去30天的地震数据，每天最多50条，最小震级3.0级")
    print("=" * 60)
    
    fetch_real_earthquake_data(days=30, limit_per_day=50, min_magnitude=3.0)
