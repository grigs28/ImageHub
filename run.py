#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: 0_imagehub.py
Description: ImageHub 统一文生图平台启动脚本（FastAPI :5569）
             自动注册 aiops 数据库
"""
import os
import sys
import logging
import subprocess
from datetime import datetime, timedelta

import netifaces
import psycopg2

from tlsutil import build_uvicorn_cmd

MODEL_ID = "imagehub/multi-image-web"
BASE_DIR = "/opt/vLLM/imagehub"
SERVICE_PORT = 5569
DB_CONFIG = {
    'host': '192.168.0.39', 'port': 5432, 'user': 'grigs',
    'password': os.environ.get('AIOPS_DB_PASSWORD', 'Slnwg123$'),
    'database': 'aiops',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_192_ips():
    start_time = datetime.now()
    while (datetime.now() - start_time) < timedelta(seconds=60):
        try:
            ips = []
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        if addr['addr'].startswith('192.168.'):
                            ips.append(addr['addr'])
            if ips:
                return ips
        except Exception as e:
            logger.warning(f"获取IP失败: {e}")
        import time
        time.sleep(1)
    return ["127.0.0.1"]


def update_process_db(status='running'):
    import time
    start = datetime.now()
    while (datetime.now() - start) < timedelta(seconds=60):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            pid = os.getpid()
            ip = get_192_ips()[0]
            cur.execute("DELETE FROM model_processes WHERE model_id=%s AND ip=%s", (MODEL_ID, ip))
            cur.execute("""INSERT INTO model_processes(model_id, ip, run_mode, pid, status)
                           VALUES (%s,%s,%s,%s,%s)""", (MODEL_ID, ip, 'service', pid, status))
            conn.commit()
            logger.info(f"数据库已更新: {MODEL_ID} {ip} pid={pid} {status}")
            return
        except Exception as e:
            logger.warning(f"数据库更新失败，重试: {e}")
            import time
            time.sleep(1)
        finally:
            if 'cur' in locals():
                cur.close()
            if 'conn' in locals():
                conn.close()


def main():
    process = None
    try:
        update_process_db('running')
        env = os.environ.copy()
        env["PYTHONPATH"] = BASE_DIR
        process = subprocess.Popen(
            build_uvicorn_cmd(
                "/opt/conda3/envs/imagegen/bin/python", SERVICE_PORT,
                os.environ.get("IMAGEHUB_TLS_CERT"),
                os.environ.get("IMAGEHUB_TLS_KEY")),
            cwd=BASE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {line}", flush=True)
        rc = process.wait()
        update_process_db('stopped')
        sys.exit(1 if rc else 0)
    except KeyboardInterrupt:
        if process:
            process.terminate()
            process.wait()
        update_process_db('stopped')
        sys.exit(0)
    except Exception as e:
        logger.error(f"异常: {e}")
        update_process_db('stopped')
        sys.exit(1)


if __name__ == '__main__':
    main()
