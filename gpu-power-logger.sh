#!/bin/bash
# 每 5 秒记录 GPU 功耗/温度/显存到日志，用于下次异常重启后追溯
LOG=/var/log/gpu-power.log
while true; do
  echo "$(date '+%F %T') $(nvidia-smi --query-gpu=index,power.draw,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader | tr '\n' '|')" >> "$LOG"
  # 限制日志大小 50MB
  if [ -f "$LOG" ] && [ $(stat -c%s "$LOG") -gt 52428800 ]; then
    tail -c 25000000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  fi
  sleep 5
done
