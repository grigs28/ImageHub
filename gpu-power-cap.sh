#!/bin/bash
# 分卡限制功耗：GPU0/1（电源A，与主板同路，长期稳定）300W；GPU2/3（电源B）250W
# 可用环境变量覆盖：GPU0_LIMIT / GPU1_LIMIT / GPU2_LIMIT / GPU3_LIMIT
L0=${GPU0_LIMIT:-300}
L1=${GPU1_LIMIT:-300}
L2=${GPU2_LIMIT:-250}
L3=${GPU3_LIMIT:-250}
nvidia-smi -i 0 -pl $L0 >/dev/null 2>&1 || true
nvidia-smi -i 1 -pl $L1 >/dev/null 2>&1 || true
nvidia-smi -i 2 -pl $L2 >/dev/null 2>&1 || true
nvidia-smi -i 3 -pl $L3 >/dev/null 2>&1 || true
echo "$(date '+%F %T') power limits: gpu0=${L0}W gpu1=${L1}W gpu2=${L2}W gpu3=${L3}W"
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
