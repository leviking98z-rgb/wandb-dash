"""wandbdash — wandb 训练曲线 dashboard(网页看真曲线, 非 ASCII)。

srv-wandb:8097。shell 出 wandb_series.py(本仓 venv, 装了 wandb)拉 entity 近期/活跃 run 的
metric 序列 → TTL 缓存(wandb API 慢, ~25s)→ 前端 SVG 真曲线(每 metric 一图, 各 run 一线)。
WANDB_API_KEY 运行时从 .clusters/.tools/mon_wandb.sh 抽(不把密钥写进本仓库)。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

_REPO = Path(__file__).resolve().parent
_SERIES = str(Path(__file__).resolve().parent / "wandb_series.py")
_MON_WANDB = os.environ.get("MON_WANDB_SH", "/root/shared/.clusters/.tools/mon_wandb.sh")
_TTL = 60
_cache: dict = {"ts": 0.0, "data": None}
_lock = threading.Lock()


def _git_rev() -> str:
    try:
        r = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


_VERSION = _git_rev()


def _api_key() -> str:
    # 运行时从 mon_wandb.sh 抽 wandb key(不落进本仓库); 也允许 env 覆盖
    k = os.environ.get("WANDB_API_KEY")
    if k:
        return k
    try:
        m = re.search(r"wandb_v1_[A-Za-z0-9_]+", open(_MON_WANDB, encoding="utf-8").read())
        return m.group(0) if m else ""
    except Exception:
        return ""


def _fetch() -> dict:
    now = time.time()
    if _cache["data"] and now - _cache["ts"] < _TTL:
        return {**_cache["data"], "cached": round(now - _cache["ts"], 1)}
    with _lock:
        now = time.time()
        if _cache["data"] and now - _cache["ts"] < _TTL:
            return {**_cache["data"], "cached": round(now - _cache["ts"], 1)}
        env = {**os.environ, "WANDB_API_KEY": _api_key(), "WANDB_SILENT": "true"}
        try:
            p = subprocess.run([sys.executable, _SERIES, "--max-runs", "8"],
                               capture_output=True, text=True, timeout=120, env=env)
            data = json.loads(p.stdout) if p.stdout.strip() else {"error": (p.stderr or "空输出")[:300], "runs": []}
        except subprocess.TimeoutExpired:
            data = {"error": "wandb 拉取超时(>120s)", "runs": []}
        except Exception as e:
            data = {"error": f"{type(e).__name__}: {e}", "runs": []}
        _cache["ts"] = now
        _cache["data"] = data
        return {**data, "cached": 0.0}


async def health(request):
    return JSONResponse({"status": "ok", "service": "wandb", "version": _VERSION})


async def api_runs(request):
    return JSONResponse(_fetch())


async def index(request):
    return HTMLResponse(_PAGE)


_PAGE = r"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>wandb · 训练曲线</title><style>
body{background:#0d1117;color:#c9d1d9;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:14px}
h1{font-size:15px;margin:0 0 4px}.dim{color:#8b949e}.spacer{flex:1}
header{display:flex;align-items:center;gap:14px;margin-bottom:10px}
button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:3px 10px;cursor:pointer}
#legend{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0;font-size:12px}
#legend span{display:inline-flex;align-items:center;gap:5px}
#legend i{width:14px;height:3px;display:inline-block;border-radius:2px}
#charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(580px,1fr));gap:14px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px}
.card h3{margin:2px 6px 6px;font-size:12px;font-weight:600}
svg text{fill:#8b949e;font-size:10px}
.st-running{color:#3fb950}.st-crashed,.st-failed,.st-killed{color:#f85149}.st-finished{color:#8b949e}
</style></head><body>
<header><h1>📈 wandb 训练曲线</h1><span class=dim id=sub></span><span class=spacer></span>
<span class=dim id=upd></span><button onclick=load()>刷新</button></header>
<div id=legend></div><div id=charts>加载中…</div>
<script>
const PAL=['#3fb950','#58a6ff','#d29922','#f85149','#bc8cff','#39c5cf','#ff7b72','#e3b341'];
function chart(metric, runs){
 const W=560,H=200,pl=52,pr=12,pt=8,pb=22;
 let xs=[],ys=[]; runs.forEach(r=>r.pts.forEach(p=>{xs.push(p[0]);ys.push(p[1])}));
 if(xs.length<2) return '';
 const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const xr=(x1-x0)||1,yr=(y1-y0)||1;
 const X=v=>pl+(v-x0)/xr*(W-pl-pr),Y=v=>H-pb-(v-y0)/yr*(H-pt-pb);
 const g=[0,.5,1].map(f=>{const y=Y(y0+yr*f);return `<line x1=${pl} y1=${y} x2=${W-pr} y2=${y} stroke=#21262d/><text x=4 y=${y+3}>${(y0+yr*f).toPrecision(3)}</text>`}).join('');
 const lines=runs.map(r=>{const d=r.pts.map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ');
   return `<polyline fill=none stroke=${r.color} stroke-width=1.4 points="${d}"/>`}).join('');
 const xl=`<text x=${pl} y=${H-6}>${x0}</text><text x=${W-pr} y=${H-6} text-anchor=end>${x1}</text>`;
 return `<div class=card><h3>${metric}</h3><svg width=${W} height=${H} viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">${g}${lines}${xl}</svg></div>`;
}
async function load(){
 try{const r=await fetch('/api/wandb/runs');const d=await r.json();
  document.getElementById('sub').textContent=(d.entity||'')+(d.error?(' · ⚠'+d.error):'');
  document.getElementById('upd').textContent='更新 '+(d.cached?d.cached+'s前':'刚刚')+' · '+(d.runs||[]).length+' run';
  const runs=d.runs||[]; const color={}; runs.forEach((r,i)=>color[r.id]=PAL[i%PAL.length]);
  document.getElementById('legend').innerHTML=runs.map(r=>`<span><i style=background:${color[r.id]}></i><span class=st-${r.state}>${r.name}</span> <span class=dim>${r.state}·${r.project}</span></span>`).join('')||'<span class=dim>无 run</span>';
  const metrics={}; runs.forEach(r=>{for(const [m,pts] of Object.entries(r.metrics||{})){(metrics[m]=metrics[m]||[]).push({name:r.name,color:color[r.id],pts})}});
  const keys=Object.keys(metrics).sort();
  document.getElementById('charts').innerHTML=keys.map(m=>chart(m,metrics[m])).join('')||'<span class=dim>暂无曲线数据(可能没在跑的 run)</span>';
 }catch(e){document.getElementById('charts').textContent='加载失败: '+e;}
}
load();setInterval(load,30000);
</script></body></html>"""


routes = [
    Route("/", index),
    Route("/health", health, methods=["GET"]),
    Route("/api/wandb/runs", api_runs, methods=["GET"]),
]

app = Starlette(routes=routes)
