#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shlex,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",required=True);ap.add_argument("--gpus",default="0")
    ap.add_argument("--port",type=int,default=8000);ap.add_argument("--cpu-offload-gb",type=float,required=True)
    ap.add_argument("--hot-experts",type=int,required=True);ap.add_argument("--max-model-len",type=int,default=32768)
    ap.add_argument("--max-num-seqs",type=int,default=1);ap.add_argument("--trace",action="store_true")
    ap.add_argument("--execute",action="store_true");a=ap.parse_args()
    model=Path(a.model_dir).resolve()
    if not (model/"model.safetensors.index.json").exists():
        print("checkpoint index missing",file=sys.stderr);return 2
    gpu_ids=[x.strip() for x in a.gpus.split(",") if x.strip()];tp=len(gpu_ids)
    env={"CUDA_VISIBLE_DEVICES":",".join(gpu_ids),"VLLM_PLE_CPU_OFFLOAD":"1",
         "VLLM_WNA16_DYNAMIC_LRU":"1","VLLM_WNA16_MIXED_VMM_HOT_CACHE":"1",
         "VLLM_WNA16_STATIC_HOT_CACHE_SIZE":str(a.hot_experts),
         "VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS":"16",
         "FENIX_TRACE":"1" if a.trace else "0","FENIX_TRACE_DIR":"/fenix-traces"}
    ranking=Path("external/runtime/qwen38/configs/static_hot_cache_rankings.json").resolve()
    if ranking.exists():env["VLLM_WNA16_STATIC_HOT_CACHE_FILE"]="/runtime/configs/static_hot_cache_rankings.json"
    cmd=["docker","run","--rm","--gpus","all","--ipc","host","--cap-add","SYS_PTRACE",
         "--ulimit","memlock=-1","--ulimit","stack=67108864",
         "-p",f"127.0.0.1:{a.port}:{a.port}",
         "-v",f"{model}:/model:ro","-v",f"{Path('external/runtime/qwen38').resolve()}:/runtime:ro",
         "-v",f"{Path('traces/raw').resolve()}:/fenix-traces"]
    for k,v in env.items():cmd += ["-e",f"{k}={v}"]
    serve=["vllm","serve","/model","--served-model-name","qwen3.8-flash-next","--host","0.0.0.0",
           "--port",str(a.port),"--tensor-parallel-size",str(tp),"--dtype","bfloat16",
           "--language-model-only","--load-format","safetensors","--safetensors-load-strategy","lazy",
           "--offload-backend","uva","--cpu-offload-gb",str(a.cpu_offload_gb),"--cpu-offload-params","experts",
           "--max-model-len",str(a.max_model_len),"--max-num-seqs",str(a.max_num_seqs),
           "--enable-chunked-prefill","--no-async-scheduling","--disable-custom-all-reduce","--trust-remote-code"]
    if tp>1:serve += ["--enable-expert-parallel","--all2all-backend","allgather_reducescatter"]
    cmd += ["fenix-qwen38:locked"]+serve
    print(json.dumps(env,indent=2));print(shlex.join(cmd))
    return subprocess.call(cmd) if a.execute else 0
if __name__=="__main__":raise SystemExit(main())
