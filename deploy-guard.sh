#!/bin/bash
# 安全重启 ImageHub：等待正在进行的生成/设计/异步任务结束后再重启
CERT=${IMAGEHUB_TLS_CERT:-/etc/imagehub/tls/imagehub.crt}
if [ -n "${IMAGEHUB_BASE_URL:-}" ]; then
  URL="${IMAGEHUB_BASE_URL%/}/api/status"
elif [ -f "$CERT" ] && curl -fsS -m 2 --cacert "$CERT" https://127.0.0.1:5569/api/version >/dev/null 2>&1; then
  URL=https://127.0.0.1:5569/api/status
else
  URL=http://127.0.0.1:5569/api/status
fi
CURL_OPTS="-fsS -m 2"
case "$URL" in
  https://*)
    if [ -f "$CERT" ]; then CURL_OPTS="$CURL_OPTS --cacert $CERT"; fi
    ;;
esac
MAX_TRIES=${IMAGEHUB_GUARD_TRIES:-300}
SLEEP_SECS=${IMAGEHUB_GUARD_SLEEP:-2}
Z=0
B=1
for i in $(seq 1 "$MAX_TRIES"); do
  B=$(curl $CURL_OPTS "$URL" | python3 -c '
import json, sys
d = json.load(sys.stdin)
v = d.get("busy", d.get("active"))
if isinstance(v, bool):
    print(int(v))
elif isinstance(v, int) and v >= 0:
    print(v)
else:
    raise SystemExit(1)
' 2>/dev/null || echo 1)
  if [ "$B" = "0" ]; then Z=$((Z+1)); else Z=0; fi
  [ "$Z" -ge 2 ] && break
  echo "有任务在跑（busy=$B，连续空闲 ${Z}/2），等待…"
  sleep "$SLEEP_SECS"
done
if [ "$Z" -lt 2 ]; then
  echo "等待超时（busy=${B:-unknown}），未重启 ImageHub"
  exit 1
fi
systemctl restart imagehub.service
echo "已重启 ImageHub"
