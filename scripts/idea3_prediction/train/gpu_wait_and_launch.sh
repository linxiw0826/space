#!/usr/bin/env bash
# gpu_wait_and_launch.sh — 轮询等到有 N 张“空”卡就自动拉起 E-16e 训练
#
# 安全说明：本脚本【从不 kill / pkill / fuser 任何进程】。它只做两件事——
#   (1) 用 `nvidia-smi --query-gpu` 只读地查各卡显存；
#   (2) 当满足条件时，启动【我们自己的】训练脚本。
# 别人的进程完全不受影响，我们只是等待空卡。
#
# 判“空”只看显存(memory.used)，不看 util —— 因为空转/挂起的卡 util=0% 但显存仍被占。
set -u

# ================= 配置(都可用 env 覆盖) =================
NEED=${NEED:-4}                     # 需要几张空卡
MEM_FREE_MB=${MEM_FREE_MB:-10000}   # 显存 used 低于这个(MB)才算“空”
INTERVAL=${INTERVAL:-30}            # 轮询间隔(秒)
STABLE=${STABLE:-1}                 # 连续几次都空才启动(1=不防抖，一够就抢)
TRAIN_SCRIPT=${TRAIN_SCRIPT:-/data2/wlx/projects/space/scripts/idea3_prediction/train/train_e16e_feedcausal.sh}
LOG=${LOG:-/data2/wlx/logs/e16e_feedcausal_launch.log}
# ========================================================

free_gpus() {  # 打印显存 used < MEM_FREE_MB 的 GPU index，一行一个
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -v t="$MEM_FREE_MB" -F', *' '($2+0) < t {print $1}'
}

echo "[waiter] 目标: $NEED 张空卡(显存<${MEM_FREE_MB}MB) | 每 ${INTERVAL}s 查一次 | 连续 ${STABLE} 次稳定才启动"
echo "[waiter] 训练脚本: $TRAIN_SCRIPT"
echo "[waiter] 本脚本只读 nvidia-smi + 启动自己的训练，绝不 kill 任何进程"
hit=0
while true; do
  mapfile -t FREE < <(free_gpus)
  n=${#FREE[@]}
  ts=$(date '+%m-%d %H:%M:%S')
  if (( n >= NEED )); then
    hit=$((hit+1))
    echo "[$ts] 空卡=[${FREE[*]}] 共 $n 张(够) 稳定 $hit/$STABLE"
    if (( hit >= STABLE )); then
      sel=("${FREE[@]:0:NEED}")
      PICK=$(IFS=,; echo "${sel[*]}")
      echo "[$ts] ✅ 锁定 GPU=$PICK 启动 E-16e (NPROC=$NEED) 日志→$LOG"
      export CUDA_VISIBLE_DEVICES="$PICK"
      export NPROC_PER_NODE="$NEED"
      bash "$TRAIN_SCRIPT" 2>&1 | tee "$LOG"
      echo "[$(date '+%m-%d %H:%M:%S')] 训练进程退出(码 ${PIPESTATUS[0]})，waiter 结束"
      exit "${PIPESTATUS[0]}"
    fi
  else
    hit=0
    echo "[$ts] 空卡=[${FREE[*]:-无}] 共 $n 张(不够 $NEED) 继续等"
  fi
  sleep "$INTERVAL"
done
