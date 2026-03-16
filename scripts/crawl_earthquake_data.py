#!/usr/bin/env python3
# 地震数据爬取脚本

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from config.config import Config

class EarthquakeCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.config = Config()
    
    def crawl_cenc(self, pages=5):
        """爬取中国地震台网数据"""
        base_url = "http://www.ceic.ac.cn/speedsearch?time=1&page={}"
        data = []
        
        for page in range(1, pages + 1):
            url = base_url.format(page)
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                table = soup.find('table', class_='news_table')
                if not table:
                    break
                
                rows = table.find_all('tr')[1:]  # 跳过表头
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 5:
                        item = {
                            'time': cols[0].text.strip(),
                            'latitude': cols[1].text.strip(),
                            'longitude': cols[2].text.strip(),
                            'depth': cols[3].text.strip(),
                            'magnitude': cols[4].text.strip(),
                            'location': cols[5].text.strip()
                        }
                        data.append(item)
                
                print(f"爬取第{page}页完成")
                time.sleep(self.config.CRAWL_DELAY)
            except Exception as e:
                print(f"爬取第{page}页失败: {e}")
                time.sleep(self.config.CRAWL_DELAY * 2)
        
        return data
    
    def process_data(self, raw_data):
        """处理爬取的数据"""
        processed_data = []
        
        for i, item in enumerate(raw_data):
            try:
                # 提取震级数值
                magnitude_str = item['magnitude']
                magnitude = float(magnitude_str.replace('M', ''))
                
                # 提取深度数值
                depth_str = item['depth']
                depth = float(depth_str.replace('km', ''))
                
                # 处理时间
                time_str = item['time']
                
                # 构建处理后的数据
                processed_item = {
                    'id': f"eq_{i+1}",
                    'name': f"{item['location']}地震",
                    'time': time_str,
                    'magnitude': magnitude,
                    'depth': depth,
                    'location': item['location'],
                    'latitude': float(item['latitude']),
                    'longitude': float(item['longitude']),
                    'intensity': self.calculate_intensity(magnitude),
                    'description': f"{time_str}在{item['location']}发生{magnitude}级地震，震源深度{depth}公里"
                }
                processed_data.append(processed_item)
            except Exception as e:
                print(f"处理数据失败: {e}")
        
        return processed_data
    
    def calculate_intensity(self, magnitude):
        """根据震级计算烈度"""
        if magnitude < 3.0:
            return "Ⅰ-Ⅱ"
        elif magnitude < 4.0:
            return "Ⅲ"
        elif magnitude < 5.0:
            return "Ⅳ-Ⅴ"
        elif magnitude < 6.0:
            return "Ⅵ-Ⅶ"
        elif magnitude < 7.0:
            return "Ⅷ-Ⅸ"
        else:
            return "Ⅹ以上"
    
    def save_data(self, data, format='csv'):
        """保存数据"""
        if format == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(self.config.EARTHQUAKE_DATA_FILE, index=False, encoding='utf-8-sig')
            print(f"数据已保存到 {self.config.EARTHQUAKE_DATA_FILE}")
        elif format == 'json':
            with open(self.config.KG_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"数据已保存到 {self.config.KG_DATA_FILE}")
    
    def run(self):
        """运行爬取流程"""
        print("开始爬取地震数据...")
        raw_data = self.crawl_cenc(pages=10)
        print(f"爬取到 {len(raw_data)} 条数据")
        
        print("处理数据...")
        processed_data = self.process_data(raw_data)
        print(f"处理后得到 {len(processed_data)} 条数据")
        
        print("保存数据...")
        self.save_data(processed_data, format='csv')
        self.save_data(processed_data, format='json')
        
        print("爬取完成！")

if __name__ == "__main__":
    crawler = EarthquakeCrawler()
    crawler.run()