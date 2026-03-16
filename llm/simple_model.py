#!/usr/bin/env python3
# 简化的大模型加载和推理

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from kg.in_memory_kg import InMemoryKG

class SimpleEarthquakeModel:
    """简化的地震知识模型"""
    
    def __init__(self):
        self.config = Config()
        self.kg = InMemoryKG()
        self.kg.run()
    
    def generate_response(self, input_text):
        """生成回复"""
        input_text = input_text.strip()
        
        # 基于规则的回复
        if "什么是地震" in input_text:
            return "地震是地球内部能量释放导致地壳震动的自然现象，通常由板块运动、断层活动等原因引起。"
        
        elif "震级" in input_text and "分类" in input_text:
            return "地震震级通常分为：微震（<3.0级）、小震（3.0-3.9级）、中震（4.0-4.9级）、强震（5.0-5.9级）、大地震（6.0-6.9级）、巨大地震（7.0-7.9级）、特大地震（≥8.0级）。"
        
        elif "深度" in input_text and "分类" in input_text:
            return "根据震源深度，地震可分为：浅源地震（<70公里）、中源地震（70-300公里）、深源地震（>300公里）。"
        
        elif "应对" in input_text or "怎么办" in input_text:
            return "地震发生时，应立即采取以下措施：1. 保持冷静，远离建筑物、玻璃窗等危险物品；2. 如果在室内，躲在桌子等坚固家具下方；3. 如果在室外，远离建筑物、电线杆等；4. 地震后迅速撤离到安全地带，避免余震伤害。"
        
        elif "多发区" in input_text or "哪些地区" in input_text:
            return "中国地震多发区主要包括：四川、云南、西藏、青海、新疆、甘肃、河北、台湾等地区，这些地区位于板块交界处或地质构造活跃区域。"
        
        elif "地震" in input_text and "多少" in input_text:
            count = len(self.kg.earthquakes)
            return f"知识图谱中目前记录了{count}条地震数据。"
        
        elif "四川宜宾" in input_text or "宜宾" in input_text:
            results = self.kg.query_earthquakes_by_region("四川宜宾")
            if results:
                eq = results[0]
                return f"四川宜宾地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到四川宜宾的地震信息。"
        
        elif "青海玉树" in input_text or "玉树" in input_text:
            results = self.kg.query_earthquakes_by_region("青海玉树")
            if results:
                eq = results[0]
                return f"青海玉树地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到青海玉树的地震信息。"
        
        elif "台湾花莲" in input_text or "花莲" in input_text:
            results = self.kg.query_earthquakes_by_region("台湾花莲")
            if results:
                eq = results[0]
                return f"台湾花莲地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到台湾花莲的地震信息。"
        
        elif "云南大理" in input_text or "大理" in input_text:
            results = self.kg.query_earthquakes_by_region("云南大理")
            if results:
                eq = results[0]
                return f"云南大理地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到云南大理的地震信息。"
        
        elif "甘肃陇南" in input_text or "陇南" in input_text:
            results = self.kg.query_earthquakes_by_region("甘肃陇南")
            if results:
                eq = results[0]
                return f"甘肃陇南地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到甘肃陇南的地震信息。"
        
        elif "新疆阿克苏" in input_text or "阿克苏" in input_text:
            results = self.kg.query_earthquakes_by_region("新疆阿克苏")
            if results:
                eq = results[0]
                return f"新疆阿克苏地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到新疆阿克苏的地震信息。"
        
        elif "西藏昌都" in input_text or "昌都" in input_text:
            results = self.kg.query_earthquakes_by_region("西藏昌都")
            if results:
                eq = results[0]
                return f"西藏昌都地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到西藏昌都的地震信息。"
        
        elif "河北唐山" in input_text or "唐山" in input_text:
            results = self.kg.query_earthquakes_by_region("河北唐山")
            if results:
                eq = results[0]
                return f"河北唐山地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到河北唐山的地震信息。"
        
        elif "广东河源" in input_text or "河源" in input_text:
            results = self.kg.query_earthquakes_by_region("广东河源")
            if results:
                eq = results[0]
                return f"广东河源地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到广东河源的地震信息。"
        
        elif "辽宁大连" in input_text or "大连" in input_text:
            results = self.kg.query_earthquakes_by_region("辽宁大连")
            if results:
                eq = results[0]
                return f"辽宁大连地震发生在{eq['time']}，震级为{eq['magnitude']}级，震源深度{eq['depth']}公里，烈度为{eq['intensity']}。"
            else:
                return "未找到辽宁大连的地震信息。"
        
        elif "强震" in input_text or "大地震" in input_text:
            results = self.kg.query_earthquakes_by_magnitude(5.0)
            if results:
                locations = [eq['location'] for eq in results]
                return f"知识图谱中记录的强震和大地震发生在：{', '.join(locations)}。"
            else:
                return "未找到强震或大地震的记录。"
        
        elif "你好" in input_text or "hi" in input_text.lower():
            return "你好！我是地震知识助手，可以帮你查询地震信息、回答地震相关问题。你可以问我关于地震的定义、震级分类、深度分类、应对措施等问题，也可以查询具体的地震信息。"
        
        else:
            return "抱歉，我暂时无法回答这个问题。你可以问我关于地震的定义、震级分类、深度分类、应对措施等问题，或者查询具体的地震信息（如四川宜宾地震、青海玉树地震等）。"

if __name__ == "__main__":
    model = SimpleEarthquakeModel()
    
    print("测试简化模型...")
    test_questions = [
        "什么是地震？",
        "震级如何分类？",
        "深度如何分类？",
        "地震发生时应该如何应对？",
        "中国哪些地区是地震多发区？",
        "四川宜宾地震的震级是多少？",
        "青海玉树地震的震级是多少？",
        "知识图谱中有多少条地震数据？",
        "你好"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        print(f"回答: {model.generate_response(question)}")