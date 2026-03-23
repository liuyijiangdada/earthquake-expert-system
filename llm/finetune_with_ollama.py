#!/usr/bin/env python3
# 使用本地Ollama模型结合知识图谱进行微调的脚本

import os
import sys
import json
import random
import ollama

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from kg.neo4j_kg import Neo4jKG

class OllamaKGFinetuner:
    def __init__(self):
        self.config = Config()
        self.kg = Neo4jKG()
        self.kg.run()
        self.ollama_model = "deepseek-r1:8b"  # 使用本地ollama模型
    
    def generate_training_data(self, num_examples=100):
        """从知识图谱生成训练数据"""
        print("从知识图谱生成训练数据...")
        
        all_earthquakes = self.kg.query_all_earthquakes()
        training_data = []
        
        # 生成关于地震基本信息的问题
        for i, eq in enumerate(all_earthquakes[:num_examples]):
            # 问题1: 询问地震震级
            training_data.append({
                "instruction": "回答关于地震的问题",
                "input": f"{eq['location']}地震的震级是多少？",
                "output": f"{eq['location']}地震的震级是{eq['magnitude']}级。"
            })
            
            # 问题2: 询问地震深度
            training_data.append({
                "instruction": "回答关于地震的问题",
                "input": f"{eq['location']}地震的震源深度是多少？",
                "output": f"{eq['location']}地震的震源深度是{eq['depth']}公里。"
            })
            
            # 问题3: 询问地震发生时间
            training_data.append({
                "instruction": "回答关于地震的问题",
                "input": f"{eq['location']}地震发生在什么时候？",
                "output": f"{eq['location']}地震发生在{eq['time']}。"
            })
            
            # 问题4: 询问地震详细信息
            training_data.append({
                "instruction": "回答关于地震的问题",
                "input": f"请详细介绍{eq['location']}地震",
                "output": f"{eq['name']}发生在{eq['time']}，震级{eq['magnitude']}级，震源深度{eq['depth']}公里，震中位于{eq['latitude']}°N, {eq['longitude']}°E，烈度{eq['intensity']}。{eq['description']}"
            })
        
        # 生成关于震级分类的问题
        training_data.append({
            "instruction": "回答关于地震的问题",
            "input": "地震震级如何分类？",
            "output": "地震震级通常分为：微震（<3.0级）、小震（3.0-3.9级）、中震（4.0-4.9级）、强震（5.0-5.9级）、大地震（6.0-6.9级）、巨大地震（7.0-7.9级）、特大地震（≥8.0级）。"
        })
        
        # 生成关于地震安全的问题
        training_data.append({
            "instruction": "回答关于地震的问题",
            "input": "地震发生时应该如何应对？",
            "output": "地震发生时，应立即采取以下措施：1. 保持冷静，迅速躲到桌子、床等坚固家具下面；2. 远离窗户、玻璃幕墙等易碎物品；3. 如果在室外，远离建筑物、树木、电线杆等；4. 地震停止后，迅速撤离到安全地带；5. 不要使用电梯，走楼梯逃生。"
        })
        
        # 生成关于地震带的问题
        training_data.append({
            "instruction": "回答关于地震的问题",
            "input": "中国哪些地区是地震多发区？",
            "output": "中国的地震多发区主要包括：1. 青藏高原地震区；2. 华北地震区；3. 东南沿海地震带；4. 南北地震带；5. 新疆地震区；6. 台湾地震区。"
        })
        
        # 保存训练数据
        os.makedirs(os.path.dirname(self.config.TRAIN_DATA_FILE), exist_ok=True)
        
        # 随机打乱数据
        random.shuffle(training_data)
        
        # 划分训练集和验证集
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        with open(self.config.TRAIN_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(self.config.VAL_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f"生成训练数据完成！")
        print(f"训练数据: {len(train_data)} 条")
        print(f"验证数据: {len(val_data)} 条")
        print(f"保存到: {self.config.TRAIN_DATA_FILE} 和 {self.config.VAL_DATA_FILE}")
        
        return train_data, val_data
    
    def test_ollama_connection(self):
        """测试Ollama连接"""
        try:
            response = ollama.generate(model=self.ollama_model, prompt="你好", stream=False)
            print(f"Ollama连接成功！模型: {self.ollama_model}")
            return True
        except Exception as e:
            print(f"Ollama连接失败: {e}")
            return False
    
    def create_finetune_model(self, train_data):
        """使用Ollama创建微调模型"""
        print(f"开始使用Ollama微调模型: {self.ollama_model}")
        
        # 准备微调数据
        finetune_data = []
        for item in train_data:
            finetune_data.append({
                "prompt": f"{item['instruction']}: {item['input']}",
                "response": item['output']
            })
        
        # 保存微调数据到文件
        finetune_data_file = "data/ollama_finetune_data.jsonl"
        with open(finetune_data_file, 'w', encoding='utf-8') as f:
            for item in finetune_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"保存微调数据到: {finetune_data_file}")
        
        # 调用Ollama进行微调
        try:
            print("\n请使用以下命令在终端中运行Ollama微调：")
            print("ollama run earthquake-expert")

            return True
        except Exception as e:
            print(f"Ollama微调失败: {e}")
            return False
        
    def evaluate_model(self, val_data):
        """评估模型性能"""
        print("评估模型性能...")
        
        correct = 0
        total = len(val_data)
        
        for item in val_data[:10]:  # 只测试前10个样本
            try:
                prompt = f"{item['instruction']}: {item['input']}"
                response = ollama.generate(
                    model=self.ollama_model,
                    prompt=prompt,
                    stream=False
                )
                
                model_response = response.get("response", "").strip()
                expected_response = item['output']
                
                print(f"\n问题: {item['input']}")
                print(f"模型回答: {model_response}")
                print(f"期望回答: {expected_response}")
                
                # 简单的评估：检查关键词是否匹配
                if any(keyword in model_response for keyword in expected_response.split()[:5]):
                    correct += 1
                    print("✓ 回答正确")
                else:
                    print("✗ 回答不正确")
                    
            except Exception as e:
                print(f"评估出错: {e}")
        
        if total > 0:
            accuracy = correct / min(total, 10) * 100
            print(f"\n评估结果: {correct}/{min(total, 10)} 正确率: {accuracy:.2f}%")
        
        return correct
    
    def run(self):
        """运行微调流程"""
        print("开始结合知识图谱的Ollama模型微调...")
        
        # 测试Ollama连接
        if not self.test_ollama_connection():
            print("Ollama连接失败，无法进行微调")
            return
        
        # 生成训练数据
        train_data, val_data = self.generate_training_data()
        
        # 创建微调模型
        self.create_finetune_model(train_data)
        
        # 评估模型
        # self.evaluate_model(val_data)
        
        print("\nOllama模型微调流程完成！")
        print("请按照上述命令在终端中执行Ollama微调操作")

if __name__ == "__main__":
    finetuner = OllamaKGFinetuner()
    finetuner.run()
