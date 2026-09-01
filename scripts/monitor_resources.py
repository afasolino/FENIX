#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,time
from pathlib import Path

def proc(path):
    try:return Path(path).read_text()
    except Exception:return None
def gpu():
    c=["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw",
       "--format=csv,noheader,nounits"]
    try:
        p=subprocess.run(c,text=True,capture_output=True,timeout=2);return p.stdout.strip().splitlines()
    except Exception:return []
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--pid",type=int)
    ap.add_argument("--interval-ms",type=int,default=100);ap.add_argument("--duration-s",type=float,default=0);a=ap.parse_args()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with out.open("w") as f:
        try:
            while True:
                r={"monotonic_ns":time.monotonic_ns(),"gpu":gpu(),"diskstats":proc("/proc/diskstats")}
                if a.pid:r["status"]=proc(f"/proc/{a.pid}/status");r["io"]=proc(f"/proc/{a.pid}/io")
                f.write(json.dumps(r)+"\n");f.flush()
                if a.duration_s and time.monotonic()-start>=a.duration_s:break
                time.sleep(a.interval_ms/1000)
        except KeyboardInterrupt:pass
if __name__=="__main__":main()
