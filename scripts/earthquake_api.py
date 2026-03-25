#!/usr/bin/env python3
# 实时地震数据获取模块

import requests
import json
import time
from datetime import datetime, timedelta

def get_recent_earthquakes():
    """获取最近24小时的地震数据"""
    # 使用USGS地震数据API（更可靠）
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    # 构建请求参数
    params = {
        "format": "geojson",
        "starttime": (datetime.utcnow() - timedelta(hours=24)).isoformat(),
        "endtime": datetime.utcnow().isoformat(),
        "limit": 50,
        "minmagnitude": 3.0
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # 处理数据
        earthquakes = []
        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [0, 0, 0])
            
            # 转换为中国时区
            time_str = datetime.fromtimestamp(properties.get("time")/1000).strftime("%Y-%m-%d %H:%M:%S")
            
            earthquake = {
                "id": f"eq_{int(time.time())}_{len(earthquakes)}",
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
            earthquakes.append(earthquake)
        
        return earthquakes
    except Exception as e:
        print(f"获取地震数据失败: {e}")
        return []

def update_earthquake_data():
    """更新地震数据到文件"""
    recent_earthquakes = get_recent_earthquakes()
    
    if recent_earthquakes:
        # 读取现有数据
        try:
            with open("data/earthquake_data.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except:
            existing_data = []
        
        # 去重（根据时间和位置）
        existing_locations = {(eq["location"], eq["time"]) for eq in existing_data}
        new_earthquakes = []
        
        for eq in recent_earthquakes:
            if (eq["location"], eq["time"]) not in existing_locations:
                new_earthquakes.append(eq)
        
        # 合并数据
        updated_data = new_earthquakes + existing_data
        
        # 保存更新后的数据
        with open("data/earthquake_data.json", "w", encoding="utf-8") as f:
            json.dump(updated_data[:1000], f, ensure_ascii=False, indent=2)
        
        print(f"更新了 {len(new_earthquakes)} 条地震数据")
        return new_earthquakes
    return []

if __name__ == "__main__":
    print("获取实时地震数据...")
    new_earthquakes = update_earthquake_data()
    print(f"成功获取 {len(new_earthquakes)} 条新地震数据")
