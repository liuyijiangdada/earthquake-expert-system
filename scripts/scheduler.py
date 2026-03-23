#!/usr/bin/env python3
# 定时任务调度器

import schedule
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg.neo4j_kg import Neo4jKG

def update_earthquake_data():
    """更新地震数据"""
    kg = Neo4jKG()
    kg.run()
    kg.update_from_realtime_data()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 执行地震数据更新")

# 每小时更新一次
schedule.every(1).hour.do(update_earthquake_data)

print("启动定时任务调度器...")
print("每小时更新一次地震数据")

# 立即执行一次更新
update_earthquake_data()

while True:
    schedule.run_pending()
    time.sleep(60)
