#!/usr/bin/env python3
# 大模型微调数据准备脚本

import pandas as pd
import json
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class FinetuneDataPreparer:
    def __init__(self):
        self.config = Config()
    
    def load_earthquake_data(self):
        """加载地震数据"""
        df = pd.read_csv(self.config.EARTHQUAKE_DATA_FILE)
        return df
    
    def generate_finetune_data(self, df):
        """生成微调数据"""
        finetune_data = []
        
        # 生成基于地震信息的问题-答案对
        for _, row in df.iterrows():
            # 问题1：地震基本信息
            q1 = f"{row['location']}在{row['time']}发生的地震震级是多少？"
            a1 = f"{row['location']}在{row['time']}发生的地震震级是{row['magnitude']}级。"
            finetune_data.append({
                "instruction": "回答关于地震的问题",
                "input": q1,
                "output": a1
            })
            
            # 问题2：地震深度
            q2 = f"{row['name']}的震源深度是多少？"
            a2 = f"{row['name']}的震源深度是{row['depth']}公里。"
            finetune_data.append({
                "instruction": "回答关于地震的问题",
                "input": q2,
                "output": a2
            })
            
            # 问题3：地震位置
            q3 = f"{row['name']}发生在哪里？"
            a3 = f"{row['name']}发生在{row['location']}，具体坐标是纬度{row['latitude']}，经度{row['longitude']}。"
            finetune_data.append({
                "instruction": "回答关于地震的问题",
                "input": q3,
                "output": a3
            })
            
            # 问题4：地震烈度
            q4 = f"{row['name']}的地震烈度是多少？"
            a4 = f"{row['name']}的地震烈度是{row['intensity']}。"
            finetune_data.append({
                "instruction": "回答关于地震的问题",
                "input": q4,
                "output": a4
            })
            
            # 问题5：地震综合信息
            q5 = f"请详细介绍{row['name']}的情况。"
            a5 = f"{row['description']}，地震发生在{row['location']}，坐标为纬度{row['latitude']}，经度{row['longitude']}，地震烈度为{row['intensity']}。"
            finetune_data.append({
                "instruction": "回答关于地震的问题",
                "input": q5,
                "output": a5
            })
        
        # 生成一些通用的地震知识问题
        common_questions = [
            {
                "instruction": "回答关于地震的问题",
                "input": "什么是地震？",
                "output": "地震是地球内部能量释放导致地壳震动的自然现象，通常由板块运动、断层活动等原因引起。"
            },
            {
                "instruction": "回答关于地震的问题",
                "input": "地震震级如何分类？",
                "output": "地震震级通常分为：微震（<3.0级）、小震（3.0-3.9级）、中震（4.0-4.9级）、强震（5.0-5.9级）、大地震（6.0-6.9级）、巨大地震（7.0-7.9级）、特大地震（≥8.0级）。"
            },
            {
                "instruction": "回答关于地震的问题",
                "input": "地震深度如何分类？",
                "output": "根据震源深度，地震可分为：浅源地震（<70公里）、中源地震（70-300公里）、深源地震（>300公里）。"
            },
            {
                "instruction": "回答关于地震的问题",
                "input": "地震发生时应该如何应对？",
                "output": "地震发生时，应立即采取以下措施：1. 保持冷静，远离建筑物、玻璃窗等危险物品；2. 如果在室内，躲在桌子等坚固家具下方；3. 如果在室外，远离建筑物、电线杆等；4. 地震后迅速撤离到安全地带，避免余震伤害。"
            },
            {
                "instruction": "回答关于地震的问题",
                "input": "中国哪些地区是地震多发区？",
                "output": "中国地震多发区主要包括：四川、云南、西藏、青海、新疆、甘肃、河北、台湾等地区，这些地区位于板块交界处或地质构造活跃区域。"
            }
        ]
        
        finetune_data.extend(common_questions)
        return finetune_data
    
    def split_data(self, data, train_ratio=0.8):
        """将数据分为训练集和验证集"""
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        val_data = data[train_size:]
        return train_data, val_data
    
    def save_data(self, train_data, val_data):
        """保存微调数据"""
        # 保存训练数据
        with open(self.config.TRAIN_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"训练数据已保存到 {self.config.TRAIN_DATA_FILE}")
        
        # 保存验证数据
        with open(self.config.VAL_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        print(f"验证数据已保存到 {self.config.VAL_DATA_FILE}")
    
    def run(self):
        """运行数据准备流程"""
        print("开始准备大模型微调数据...")
        
        # 加载地震数据
        df = self.load_earthquake_data()
        print(f"加载了 {len(df)} 条地震数据")
        
        # 生成微调数据
        finetune_data = self.generate_finetune_data(df)
        print(f"生成了 {len(finetune_data)} 条微调数据")
        
        # 分割数据
        train_data, val_data = self.split_data(finetune_data)
        print(f"训练集：{len(train_data)} 条，验证集：{len(val_data)} 条")
        
        # 保存数据
        self.save_data(train_data, val_data)
        
        print("大模型微调数据准备完成！")

if __name__ == "__main__":
    preparer = FinetuneDataPreparer()
    preparer.run()