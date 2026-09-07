#!/usr/bin/env bash
# ============================================================================
#  AutoDL 上从零跑通「地震应急问答」评测 —— 纯基座外部对照基线（路径 B）
#  适用：AutoDL GPU 实例（PyTorch 镜像，CUDA 12.x，24G+ 显存，如 4090/3090）
#  目标：用原生 Qwen2.5-7B-Instruct（不加载 LoRA）在 60 题上跑 B0，
#        拿到「外部对照基线」分数，补论文里「跟别人比」的缺口。
#
#  用法（在 AutoDL 实例终端里执行本脚本）：
#    bash autodl_run_eval.sh
#  建议先用 tmux 跑，避免 SSH 断开中断：
#    tmux new -s eval && bash autodl_run_eval.sh
#    # Ctrl+B 再按 D 脱离；回来： tmux attach -s eval
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. 路径约定（按需修改）
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/root/biyelunwen}"          # 代码同步到实例后的根目录
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"  # 基座模型存放（数据盘）
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/eval_out}"   # 评测结果输出
PYTHON_BIN="python3"

echo "===== [0] 路径 ====="
echo "PROJECT_DIR = $PROJECT_DIR"
echo "MODEL_DIR   = $MODEL_DIR"
echo "OUTPUT_DIR  = $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 1. 进入项目并安装最小依赖
#    （评测链路只用到 torch/transformers/peft + requests/dotenv，无需 neo4j/flask/langgraph）
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"

echo "===== [1] 检查/安装依赖 ====="
$PYTHON_BIN - <<'PY'
import importlib.util as u
need = ["torch","transformers","peft","accelerate","sentencepiece","huggingface_hub","requests","dotenv","pandas","numpy"]
miss = [p for p in need if u.find_spec(p) is None]
print("缺失:", miss if miss else "无（全部已装）")
PY

# 若上面的「缺失」非空，或 torch 没有 CUDA，则安装
if ! $PYTHON_BIN -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>/dev/null; then
  echo ">>> 安装 torch(CUDA12.1) + 轻量依赖 ..."
  pip install --quiet --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1
  pip install --quiet transformers==4.46.3 peft==0.13.2 accelerate==0.32.1 \
    sentencepiece huggingface_hub requests python-dotenv pandas numpy
else
  echo ">>> torch+CUDA 已就绪，仅补装轻量依赖 ..."
  pip install --quiet transformers==4.46.3 peft==0.13.2 accelerate==0.32.1 \
    sentencepiece huggingface_hub requests python-dotenv pandas numpy
fi

# ---------------------------------------------------------------------------
# 2. 下载基座模型 Qwen2.5-7B-Instruct（走 HF 镜像，国内快）
#    下载到数据盘 MODEL_DIR；脚本用 --base-model-path 直接指向它。
# ---------------------------------------------------------------------------
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "===== [2] 模型已存在，跳过下载 ====="
else
  echo "===== [2] 下载 Qwen2.5-7B-Instruct → $MODEL_DIR ====="
  mkdir -p "$(dirname "$MODEL_DIR")"
  export HF_ENDPOINT="https://hf-mirror.com"
  # 若 hf-mirror 不通，可改用 ModelScope：
  #   pip install modelscope
  #   modelscope download --model qwen/Qwen2.5-7B-Instruct --local_dir "$MODEL_DIR"
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir "$MODEL_DIR" --local-dir-use-symlink False
fi

# ---------------------------------------------------------------------------
# 3. 跑评测：原生 7B（--no-lora）+ B0 配置（无 KG / 无 RAG / 动态源保留）
#    --run-llm        真正调用模型生成（默认只组装上下文）
#    --no-lora        跳过缺失的 7B LoRA，纯基座
#    --base-model-path 指向刚下载的本地基座
#    --baselines B0   只跑外部对照基线（60 题 = 3 阶段 × 20）
# ---------------------------------------------------------------------------
echo "===== [3] 运行评测（B0 纯基座外部对照）====="
mkdir -p "$OUTPUT_DIR"

$PYTHON_BIN scripts/run_ablation_eval_v2.py \
  --run-llm \
  --no-lora \
  --base-model-path "$MODEL_DIR" \
  --baselines B0 \
  --per-phase 20 \
  --output "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/eval_b0_run.log"

echo ""
echo "===== [4] 结果汇总 ====="
echo "详细 JSON : $OUTPUT_DIR/ablation_results_extended.json"
echo "汇总 JSON : $OUTPUT_DIR/table_extended_summary.json"
$PYTHON_BIN - <<'PY'
import json, pathlib
p = pathlib.Path("/root/autodl-tmp/eval_out") / "table_extended_summary.json"
if p.exists():
    s = json.loads(p.read_text(encoding="utf-8"))
    for bid, v in s.items():
        print(f"{bid}: legacy事实一致性={v['legacy']['factual_accuracy_pct']}% | "
              f"normalized={v['normalized']['factual_accuracy_pct']}% | n={v['n']}")
else:
    print("汇总文件尚未生成，请检查上面的运行日志。")
PY

echo ""
echo "【说明】这是原生 Qwen2.5-7B（无 LoRA）在 B0 配置下的外部对照分数。"
echo "若想进一步对比「纯基座 + KG/RAG」(B0/B1/B2/B3 全用纯基座)，"
echo "把上面 --baselines B0 改为 --baselines B0,B1,B2,B3，并先装句向量："
echo "  pip install sentence-transformers   # B2/B3 走 embedding RAG，否则回退关键词 RAG"
