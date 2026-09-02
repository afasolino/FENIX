#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,queue,threading,time,uuid
from pathlib import Path
import requests

def pct(xs,p):
    if not xs:return None
    ys=sorted(xs);k=(len(ys)-1)*p;f=int(k);c=min(f+1,len(ys)-1)
    return ys[f] if f==c else ys[f]*(c-k)+ys[c]*(k-f)

def run_one(url,model,prompt,max_tokens,temp,rid):
    t0=time.perf_counter_ns();first=None;usage=None
    payload={"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,
             "temperature":temp,"stream":True,"stream_options":{"include_usage":True}}
    with requests.post(url,json=payload,stream=True,timeout=3600) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):continue
            data=raw[6:]
            if data=="[DONE]":break
            obj=json.loads(data)
            if obj.get("usage"):usage=obj["usage"]
            ch=obj.get("choices") or []
            delta=(ch[0].get("delta") or {}) if ch else {}
            if first is None and (delta.get("content") or delta.get("reasoning_content")):first=time.perf_counter_ns()
    t1=time.perf_counter_ns();u=usage or {};ct=u.get("completion_tokens");pt=u.get("prompt_tokens")
    return {"request_id":rid,"start_ns":t0,"first_token_ns":first,"end_ns":t1,
            "prompt_tokens":pt,"completion_tokens":ct,
            "ttft_ms":None if first is None else (first-t0)/1e6,
            "e2e_ms":(t1-t0)/1e6,
            "tpot_ms":None if first is None or not ct or ct<2 else (t1-first)/1e6/(ct-1),
            "decode_tokens_s":None if first is None or not ct or ct<2 else (ct-1)/((t1-first)/1e9)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--url",default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model",default="qwen3.8-flash-next");ap.add_argument("--out",required=True)
    ap.add_argument("--requests",type=int,default=20);ap.add_argument("--concurrency",type=int,default=1)
    ap.add_argument("--max-tokens",type=int,default=256);ap.add_argument("--temperature",type=float,default=0)
    ap.add_argument("--prompt",default="Explain deterministic sparse lookup prefetch during autoregressive decoding.")
    a=ap.parse_args();jobs=queue.Queue();results=[];lock=threading.Lock()
    for i in range(a.requests):jobs.put((i,f"fenix-{i:06d}-{uuid.uuid4().hex[:8]}"))
    def worker():
        while True:
            try:i,rid=jobs.get_nowait()
            except queue.Empty:return
            try:r=run_one(a.url,a.model,a.prompt,a.max_tokens,a.temperature,rid);r.update(ordinal=i,concurrency=a.concurrency)
            except Exception as e:r={"request_id":rid,"ordinal":i,"concurrency":a.concurrency,"error":repr(e)}
            with lock:results.append(r)
            jobs.task_done()
    t0=time.perf_counter_ns();ts=[threading.Thread(target=worker) for _ in range(a.concurrency)]
    [t.start() for t in ts];[t.join() for t in ts];t1=time.perf_counter_ns()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w") as f:
        for r in sorted(results,key=lambda x:x["ordinal"]):f.write(json.dumps(r)+"\n")
    good=[r for r in results if "error" not in r];wall=(t1-t0)/1e9
    summ={"requests":len(results),"success":len(good),"concurrency":a.concurrency,"wall_s":wall,
          "aggregate_completion_tokens_s":sum(r.get("completion_tokens") or 0 for r in good)/wall if wall else None,
          "aggregate_prompt_tokens_s":sum(r.get("prompt_tokens") or 0 for r in good)/wall if wall else None}
    for fld in ("ttft_ms","tpot_ms","e2e_ms"):
        xs=[r[fld] for r in good if r.get(fld) is not None]
        for name,p in (("p50",.5),("p95",.95),("p99",.99)):summ[f"{fld}_{name}"]=pct(xs,p)
    Path(str(out)+".summary.json").write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2))
if __name__=="__main__":main()
