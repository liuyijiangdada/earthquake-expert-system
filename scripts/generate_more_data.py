#!/usr/bin/env python3
# 生成更多地震数据脚本

import json
import random
import time
from datetime import datetime, timedelta

# 中国主要地区
regions = [
    "四川", "云南", "青海", "西藏", "新疆", "甘肃", "河北", "台湾", "广东", "辽宁",
    "北京", "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北",
    "湖南", "广西", "海南", "重庆", "贵州", "陕西", "吉林", "黑龙江", "内蒙古", "宁夏"
]

# 生成地震数据
def generate_earthquake_data(num_records=1000):
    """生成地震数据"""
    earthquakes = []
    
    # 读取现有数据
    try:
        with open("data/earthquake_data.json", "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except:
        existing_data = []
    
    # 计算需要生成的记录数
    existing_count = len(existing_data)
    need_generate = num_records - existing_count
    
    if need_generate <= 0:
        print(f"已有 {existing_count} 条数据，已达到或超过目标数量 {num_records}")
        return existing_data
    
    print(f"需要生成 {need_generate} 条数据")
    
    # 生成新数据
    for i in range(need_generate):
        # 随机选择地区
        region = random.choice(regions)
        cities = ["成都", "宜宾", "大理", "玉树", "拉萨", "乌鲁木齐", "兰州", "唐山", "台北", "广州", "沈阳"]
        location = f"{region}{random.choice(cities)}"
        
        # 随机生成时间（过去30天内）
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        earthquake_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        time_str = earthquake_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 随机生成震级（3.0-7.0级）
        magnitude = round(random.uniform(3.0, 7.0), 1)
        
        # 随机生成深度（5-50公里）
        depth = random.randint(5, 50)
        
        # 随机生成经纬度（基于地区）
        if region == "四川":
            latitude = round(random.uniform(26.0, 34.0), 3)
            longitude = round(random.uniform(97.0, 109.0), 3)
        elif region == "云南":
            latitude = round(random.uniform(21.0, 29.0), 3)
            longitude = round(random.uniform(97.0, 106.0), 3)
        elif region == "青海":
            latitude = round(random.uniform(31.0, 39.0), 3)
            longitude = round(random.uniform(89.0, 103.0), 3)
        elif region == "西藏":
            latitude = round(random.uniform(26.0, 36.0), 3)
            longitude = round(random.uniform(78.0, 99.0), 3)
        elif region == "新疆":
            latitude = round(random.uniform(34.0, 49.0), 3)
            longitude = round(random.uniform(73.0, 96.0), 3)
        else:
            # 其他地区的默认范围
            latitude = round(random.uniform(20.0, 53.0), 3)
            longitude = round(random.uniform(73.0, 135.0), 3)
        
        # 生成烈度
        if magnitude < 4.0:
            intensity = "IV度"
        elif magnitude < 5.0:
            intensity = "V度"
        elif magnitude < 6.0:
            intensity = "VI度"
        elif magnitude < 7.0:
            intensity = "VII度"
        else:
            intensity = "VIII度"
        
        # 生成描述
        description = f"震级{magnitude}级地震，震感{intensity}，{depth}公里深度"
        
        # 生成ID
        earthquake_id = f"eq_{int(time.time())}_{i}"
        
        # 创建地震数据
        earthquake = {
            "id": earthquake_id,
            "name": f"{location}地震",
            "location": location,
            "time": time_str,
            "magnitude": magnitude,
            "depth": depth,
            "latitude": latitude,
            "longitude": longitude,
            "intensity": intensity,
            "description": description
        }
        
        earthquakes.append(earthquake)
        
        # 每生成100条数据打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1} 条数据")
    
    # 合并数据
    updated_data = existing_data + earthquakes
    
    # 限制数据数量为1000条
    if len(updated_data) > num_records:
        updated_data = updated_data[:num_records]
    
    # 保存数据
    with open("data/earthquake_data.json", "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据生成完成！")
    print(f"总数据量：{len(updated_data)} 条")
    
    return updated_data

if __name__ == "__main__":
    print("开始生成地震数据...")
    generate_earthquake_data(1000)
