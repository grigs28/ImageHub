#!/bin/bash
# 安全重启 ImageHub：等待正在进行的生成任务结束后再重启，避免打断对比任务
URL=http://127.0.0.1:5569/api/status
for i in $(seq 1 300); do
  A=$(curl -s -m 2 "$URL" | python3 -c "import json,sys;print(json.load(sys.stdin).get('active',0))" 2>/dev/null || echo 0)
  [ "$A" = "0" ] && [ "$i" -gt 1 ] && break
  [ "$i" -eq 1 ] && [ "$A" = "0" ] && break
  echo "有 $A 个任务在跑，等待…"; sleep 2
done
systemctl restart imagehub.service
echo "已重启 ImageHub"
