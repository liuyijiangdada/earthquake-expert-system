#!/bin/bash
# 启动脚本

echo "启动地震知识图谱和大模型应用..."

# 检查是否安装了依赖
if [ ! -f "requirements.txt" ]; then
    echo "错误：requirements.txt 文件不存在"
    exit 1
fi

# 启动Flask应用
echo "启动Flask应用..."
python app.py