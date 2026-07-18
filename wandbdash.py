"""wandbdash — wandb 训练曲线 dashboard(workspace 式: 左侧栏选 run + 右主区画曲线)。

srv-wandb:8097。shell 出 wandb_series.py(本仓 venv, 装了 wandb)拉 entity 近期/活跃 run 的
metric 序列(点=[step,val,ts] + config/summary/runtime/created)→ TTL 缓存 → 前端 SVG 曲线 + 交互。
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
_hist_cache: dict = {}   # (project,id) -> (ts, run_dict); 单 run 曲线按需拉的缓存
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
            p = subprocess.run([sys.executable, _SERIES],
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


async def api_history(request):
    """按需拉单个 run 的曲线 + config + summary(列表模式为省时不带这些)。"""
    proj = request.query_params.get("project", "")
    rid = request.query_params.get("id", "")
    if not proj or not rid:
        return JSONResponse({"error": "缺 project/id"}, status_code=400)
    key = (proj, rid)
    now = time.time()
    hit = _hist_cache.get(key)
    if hit and now - hit[0] < 300:
        return JSONResponse({**hit[1], "cached": round(now - hit[0], 1)})
    env = {**os.environ, "WANDB_API_KEY": _api_key(), "WANDB_SILENT": "true"}
    try:
        p = subprocess.run([sys.executable, _SERIES, "--run-project", proj, "--run-id", rid],
                           capture_output=True, text=True, timeout=60, env=env)
        d = json.loads(p.stdout) if p.stdout.strip() else {"error": (p.stderr or "空输出")[:200]}
    except Exception as e:
        d = {"error": f"{type(e).__name__}: {e}"}
    run = d.get("run") or {"metrics": {}, "config": {}, "summary": {}, "error": d.get("error")}
    _hist_cache[key] = (now, run)
    return JSONResponse({**run, "cached": 0.0})


async def index(request):
    return HTMLResponse(_PAGE)


_PAGE = r"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>wandb · workspace</title><style>
:root{--bg:#0d1117;--card:#161b22;--fg:#c9d1d9;--dim:#8b949e;--line:#30363d;--sel:#1f2733}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:10px}
h1{font-size:15px;margin:0}.dim{color:var(--dim)}.spacer{flex:1}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px}
button{background:#21262d;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer}
button:hover{border-color:#58a6ff}
#wrap{display:flex;gap:10px;align-items:flex-start}
#side{width:280px;flex:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;position:sticky;top:8px;max-height:calc(100vh - 20px);overflow:auto}
#main{flex:1;min-width:0}
#filter{width:100%;background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;margin-bottom:6px;font:inherit}
.run{display:flex;align-items:flex-start;gap:6px;padding:5px 4px;border-radius:6px;cursor:default}
.run:hover{background:var(--sel)}
.run i{width:12px;height:12px;border-radius:3px;flex:none;margin-top:2px}
.run .nm{color:#58a6ff;cursor:pointer;word-break:break-all}
.run .props{color:var(--dim);font-size:10.5px}
.st-running{color:#3fb950}.st-crashed,.st-failed,.st-killed{color:#f85149}.st-finished{color:#8b949e}
#pager{display:flex;align-items:center;gap:8px;justify-content:center;margin-top:8px;font-size:11px}
#pager button{padding:1px 8px}
#ctl{display:flex;align-items:center;gap:16px;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin-bottom:10px;font-size:12px}
#ctl label{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
#charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.card h3{margin:2px 6px 4px;font-size:12px;font-weight:600}
svg text{fill:var(--dim);font-size:10px}
#tip{position:fixed;pointer-events:none;background:#1f2733;border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:11px;display:none;z-index:9;max-width:260px;box-shadow:0 2px 8px #0008}
#tip b{color:var(--fg)}
details{margin-top:12px}summary{cursor:pointer;color:#58a6ff;font-weight:600}
#menu{display:none}
#sideback{display:none;position:fixed;inset:0;background:#0008;z-index:29}
#sideback.on{display:block}
table{border-collapse:collapse;width:100%;font-size:11px;margin-top:6px}
th,td{border:1px solid var(--line);padding:3px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{cursor:pointer;background:var(--sel);position:sticky;top:0}
td.name{color:#58a6ff;cursor:pointer}
#tblwrap{overflow:auto;max-height:340px}
#modal{position:fixed;inset:0;background:#000a;display:none;align-items:center;justify-content:center;z-index:20}
#modalbox{background:var(--card);border:1px solid var(--line);border-radius:10px;max-width:90vw;max-height:80vh;overflow:auto;padding:16px;min-width:300px}
.sec{margin:14px 0 4px;font-size:13px;font-weight:600;color:#58a6ff}
@media(max-width:640px){
 #menu{display:inline-block}
 #side{position:fixed;top:0;left:0;height:100vh;width:82vw;max-width:320px;max-height:none;z-index:30;transform:translateX(-102%);transition:transform .2s;border-radius:0}
 #side.open{transform:none;box-shadow:2px 0 16px #000a}
}
</style></head><body>
<header><button id=menu onclick=toggleSide()>☰ 运行</button><h1>📈 wandb workspace</h1><span class=dim id=sub></span><span class=spacer></span>
<span class=dim id=upd></span><button onclick=load()>刷新</button></header>
<div id=sideback onclick=toggleSide()></div>
<div id=wrap>
 <aside id=side>
  <input id=filter placeholder="过滤 run(名/状态/项目)…" oninput="page=0;render()">
  <div id=runlist></div>
  <div id=pager></div>
 </aside>
 <main id=main>
  <div id=ctl>
   <label>平滑 <input type=range id=sm min=0 max=0.95 step=0.05 value=0 oninput=apply()><span id=smv>0.00</span></label>
   <label><input type=checkbox id=logy onchange=apply()> 对数 Y</label>
   <label>X:
     <label><input type=radio name=xm id=xstep checked onchange=apply()> step</label>
     <label><input type=radio name=xm id=xtime onchange=apply()> 时间</label></label>
   <button onclick=resetZoom()>复位缩放</button>
   <span class=dim>拖动框选放大 · 双击复位 · 悬停看值 · 侧栏勾选 run</span>
  </div>
  <div id=charts>加载中…</div>
  <details id=tblsec><summary>Runs 表格(点表头排序 · 点 run 看 config/summary)</summary><div id=tblwrap></div></details>
 </main>
</div>
<div id=tip></div>
<div id=modal onclick="if(event.target.id=='modal')this.style.display='none'"><div id=modalbox></div></div>
<script>
const PAL=['#3fb950','#58a6ff','#d29922','#f85149','#bc8cff','#39c5cf','#ff7b72','#e3b341'];
let RAW={runs:[]}; const hidden=new Set(); const zoom={}; const COLOR={}; let inited=false;
let smooth=0,logY=false,xmode='step'; let sortCol=null,sortDir=1; let page=0; const PS=10;
const q=id=>document.getElementById(id), TIP=()=>q('tip');
const nMetrics=r=>Object.keys(r&&r.metrics||{}).length;
function fmtNum(v){if(v==null||isNaN(v))return '—';const a=Math.abs(v);if(a!==0&&(a<1e-3||a>=1e5))return v.toExponential(2);return (Math.round(v*1e4)/1e4).toString();}
function fmtX(x){if(xmode==='time'){const d=new Date(x);return isNaN(d)?'-':d.toLocaleTimeString('zh',{hour12:false});}return Math.round(x).toString();}
function fmtRt(s){if(s==null)return '—';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);if(h)return h+'h'+m+'m';if(m)return m+'m'+(s%60)+'s';return s+'s';}
function getX(p){return xmode==='time'?(p[2]?p[2]*1000:NaN):p[0];}
function ema(pts,a){if(!(a>0))return pts.map(p=>[p.x,p.y]);let s=null,o=[];for(const p of pts){s=s==null?p.y:a*s+(1-a)*p.y;o.push([p.x,s]);}return o;}

function apply(){smooth=parseFloat(q('sm').value)||0;q('smv').textContent=smooth.toFixed(2);logY=q('logy').checked;xmode=q('xstep').checked?'step':'time';render();}
function resetZoom(){for(const k in zoom)delete zoom[k];render();}
async function toggleRun(id){if(hidden.has(id)){hidden.delete(id);await ensureHistory(id);}else{hidden.add(id);}render();}
function toggleSide(){const s=q('side');const on=s.classList.toggle('open');q('sideback').classList.toggle('on',on);}

async function load(){try{const r=await fetch('/api/wandb/runs');const nw=await r.json();
 // 合并: 保留已按需拉到的曲线/config, 避免 30s 轮询把它们冲掉
 const prev={};(RAW.runs||[]).forEach(x=>prev[x.id]=x);
 (nw.runs||[]).forEach(x=>{const o=prev[x.id];if(o){if(!nMetrics(x)&&nMetrics(o))x.metrics=o.metrics;if(o._loaded){x._loaded=true;if(o.config&&Object.keys(o.config).length)x.config=o.config;if(o.summary&&Object.keys(o.summary).length&&!Object.keys(x.summary||{}).length)x.summary=o.summary;}}});
 RAW=nw;
 const runs=RAW.runs||[];
 q('sub').textContent=(RAW.entity||'')+(RAW.error?(' · ⚠'+RAW.error):'');
 const nc=runs.filter(r=>nMetrics(r)).length;
 q('upd').textContent='更新 '+(RAW.cached?RAW.cached+'s前':'刚刚')+' · '+runs.length+' run · '+nc+' 有曲线';
 runs.forEach((r,i)=>{if(!COLOR[r.id])COLOR[r.id]=PAL[i%PAL.length];});
 if(!inited){inited=true;
   // 默认只画在跑的(没有则画前几个有曲线的), 其余 run 全在侧栏列表里可勾选
   const running=runs.filter(r=>r.state==='running');
   const show=(running.length?running:runs.filter(r=>nMetrics(r)).slice(0,6)).map(r=>r.id);
   const ss=new Set(show); runs.forEach(r=>{if(!ss.has(r.id))hidden.add(r.id);});
 }
 render();
}catch(e){q('charts').textContent='加载失败: '+e;}}

// 按需拉某个 run 的曲线 + config(列表模式为省时不带)
async function ensureHistory(id){
 const r=(RAW.runs||[]).find(x=>x.id===id); if(!r||r._loaded||nMetrics(r))return;
 r._loading=true; render();
 try{const res=await fetch('/api/wandb/history?project='+encodeURIComponent(r.project)+'&id='+encodeURIComponent(r.id));
   const d=await res.json();
   if(d){if(d.metrics)r.metrics=d.metrics;
     if(d.config&&Object.keys(d.config).length)r.config=d.config;
     if(d.summary&&Object.keys(d.summary).length)r.summary=d.summary;}
 }catch(e){}
 r._loading=false; r._loaded=true;
}

function render(){
 const runs=RAW.runs||[];
 // 侧栏: 过滤 + 分页
 const f=(q('filter').value||'').toLowerCase();
 const filt=runs.filter(r=>!f||(r.name+' '+r.state+' '+r.project).toLowerCase().includes(f));
 const pages=Math.max(1,Math.ceil(filt.length/PS)); if(page>=pages)page=pages-1; if(page<0)page=0;
 const paged=filt.slice(page*PS,page*PS+PS);
 q('runlist').innerHTML=paged.map(r=>{
   const off=hidden.has(r.id);
   const mark=r._loading?' <span class=dim>⏳</span>':(nMetrics(r)?' <span title="有曲线">📈</span>':'');
   return `<div class=run><input type=checkbox ${off?'':'checked'} onchange="toggleRun('${r.id}')"><i style="background:${COLOR[r.id]}"></i><div><span class=nm onclick="detail('${r.id}')">${r.name}</span>${mark}<div class=props><span class="st-${r.state}">${r.state}</span> · ⏱${fmtRt(r.runtime)} · #${(r.id||'').slice(0,8)} · ${r.project}</div></div></div>`;
 }).join('')||'<div class=dim style=padding:8px>无匹配 run</div>';
 q('pager').innerHTML=filt.length?`<button onclick="page--;render()" ${page<=0?'disabled':''}>‹</button><span class=dim>${page*PS+1}–${Math.min((page+1)*PS,filt.length)} / ${filt.length}${f?' (筛后)':''}</span><button onclick="page++;render()" ${page>=pages-1?'disabled':''}>›</button>`:'';
 // 主区: 图表(画所有勾选=未 hidden 的 run, 不受分页/过滤影响)
 const vis=runs.filter(r=>!hidden.has(r.id));
 const metrics={};vis.forEach(r=>{for(const m in (r.metrics||{}))(metrics[m]=metrics[m]||[]).push(r);});
 const keys=Object.keys(metrics).sort();
 const cont=q('charts');cont.innerHTML='';
 if(!keys.length)cont.innerHTML='<span class=dim>暂无曲线 —— 侧栏勾选 run 即按需加载其曲线(默认只画在跑的)</span>';
 keys.forEach(m=>{const c=document.createElement('div');c.className='card';cont.appendChild(c);drawChart(c,m,metrics[m]);});
 buildTable(runs);
}

function drawChart(card,metric,runsForMetric){
 const W=560,H=210,pl=54,pr=12,pt=10,pb=24;
 let series=runsForMetric.map(r=>{
   let pts=(r.metrics[metric]||[]).map(p=>({x:getX(p),y:p[1]})).filter(p=>isFinite(p.x)&&isFinite(p.y)).sort((a,b)=>a.x-b.x);
   const z=zoom[metric];if(z)pts=pts.filter(p=>p.x>=z[0]&&p.x<=z[1]);
   return {name:r.name,color:COLOR[r.id],pts:ema(pts,smooth)};
 }).filter(s=>s.pts.length>=1);
 let xs=[],ys=[];series.forEach(s=>s.pts.forEach(p=>{xs.push(p[0]);ys.push(p[1]);}));
 if(xs.length<2){card.innerHTML=`<h3>${metric}</h3><div class=dim style=padding:16px>无点</div>`;return;}
 const x0=Math.min(...xs),x1=Math.max(...xs);
 let yv=ys.slice();if(logY){const pos=yv.filter(v=>v>0);yv=pos.length?pos.map(v=>Math.log10(v)):yv;}
 let y0=Math.min(...yv),y1=Math.max(...yv);if(y0===y1){y0-=1;y1+=1;}
 const xr=(x1-x0)||1,yr=(y1-y0)||1;
 const X=v=>pl+(v-x0)/xr*(W-pl-pr);
 const YV=v=>logY?(v>0?Math.log10(v):y0):v;
 const Y=v=>H-pb-(YV(v)-y0)/yr*(H-pt-pb);
 let grid='';for(let i=0;i<=3;i++){const t=y0+yr*i/3;const py=H-pb-(t-y0)/yr*(H-pt-pb);grid+=`<line x1="${pl}" y1="${py.toFixed(1)}" x2="${W-pr}" y2="${py.toFixed(1)}" stroke="#21262d"></line><text x="6" y="${(py+3).toFixed(1)}">${fmtNum(logY?Math.pow(10,t):t)}</text>`;}
 const lines=series.map(s=>`<polyline fill="none" stroke="${s.color}" stroke-width="1.5" points="${s.pts.map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ')}"></polyline>`).join('');
 const xl=`<text x="${pl}" y="${H-6}">${fmtX(x0)}</text><text x="${W-pr}" y="${H-6}" text-anchor="end">${fmtX(x1)}</text>`;
 card.innerHTML=`<h3>${metric}${zoom[metric]?' <span class=dim style=font-weight:400>· 已缩放(双击复位)</span>':''}</h3>
  <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;touch-action:none">${grid}${lines}${xl}
  <line class="cross" y1="${pt}" y2="${H-pb}" stroke="#8b949e" stroke-dasharray="3,3" style="display:none"></line>
  <rect class="band" y="${pt}" height="${H-pt-pb}" fill="#58a6ff33" style="display:none"></rect>
  <rect class="ov" x="${pl}" y="${pt}" width="${W-pl-pr}" height="${H-pt-pb}" fill="transparent" style="cursor:crosshair"></rect></svg>`;
 const svg=card.querySelector('svg'),cross=svg.querySelector('.cross'),band=svg.querySelector('.band'),ov=svg.querySelector('.ov');
 const evX=ev=>{const r=svg.getBoundingClientRect();const sx=(ev.clientX-r.left)/r.width*W;return{sx,dx:x0+(sx-pl)/(W-pl-pr)*xr};};
 let drag=null;
 ov.addEventListener('pointermove',ev=>{const {sx,dx}=evX(ev);
   if(drag!=null){const a=X(drag);band.setAttribute('x',Math.min(a,sx));band.setAttribute('width',Math.abs(sx-a));band.style.display='';cross.style.display='none';TIP().style.display='none';return;}
   cross.setAttribute('x1',sx);cross.setAttribute('x2',sx);cross.style.display='';
   let html=`<b>${metric}</b> · ${xmode==='time'?'时间':'step'} ${fmtX(dx)}<br>`;
   series.forEach(s=>{let best=null,bd=1e18;for(const p of s.pts){const d=Math.abs(p[0]-dx);if(d<bd){bd=d;best=p;}}if(best)html+=`<span style=color:${s.color}>■</span> ${s.name}: <b>${fmtNum(best[1])}</b><br>`;});
   const t=TIP();t.innerHTML=html;t.style.display='';t.style.left=Math.min(ev.clientX+12,innerWidth-t.offsetWidth-8)+'px';t.style.top=(ev.clientY+12)+'px';});
 ov.addEventListener('pointerdown',ev=>{ov.setPointerCapture(ev.pointerId);drag=evX(ev).dx;});
 ov.addEventListener('pointerup',ev=>{if(drag==null)return;const dx=evX(ev).dx;const a=Math.min(drag,dx),b=Math.max(drag,dx);drag=null;band.style.display='none';if(b-a>xr*0.02){zoom[metric]=[a,b];render();}});
 ov.addEventListener('pointerleave',()=>{cross.style.display='none';TIP().style.display='none';});
 svg.addEventListener('dblclick',()=>{if(zoom[metric]){delete zoom[metric];render();}});
}

function buildTable(runs){
 const sumKeys=[...new Set(runs.flatMap(r=>Object.keys(r.summary||{})))].sort();
 const cols=[{k:'name',t:'run'},{k:'state',t:'state'},{k:'rt',t:'runtime'},{k:'project',t:'project'},...sumKeys.map(k=>({k:'sum:'+k,t:k}))];
 const val=(r,c)=>c.k==='name'?r.name:c.k==='state'?r.state:c.k==='rt'?(r.runtime||0):c.k==='project'?r.project:(r.summary||{})[c.k.slice(4)];
 let rs=runs.slice();
 if(sortCol!=null){const c=cols[sortCol];rs.sort((a,b)=>{let x=val(a,c),y=val(b,c);if(typeof x==='number'&&typeof y==='number')return (x-y)*sortDir;return String(x).localeCompare(String(y))*sortDir;});}
 const th=cols.map((c,i)=>`<th onclick=sortBy(${i})>${c.t}${sortCol===i?(sortDir>0?' ▲':' ▼'):''}</th>`).join('');
 const tr=rs.map(r=>'<tr>'+cols.map((c,i)=>{let v=val(r,c);if(i===0)return `<td class=name onclick="detail('${r.id}')"><span style=color:${COLOR[r.id]}>■</span> ${v}</td>`;if(c.k==='rt')return `<td>${fmtRt(r.runtime)}</td>`;if(c.k.startsWith('sum:'))v=typeof v==='number'?fmtNum(v):(v??'—');if(c.k==='state')return `<td class="st-${r.state}">${v}</td>`;return `<td>${v}</td>`;}).join('')+'</tr>').join('');
 q('tblwrap').innerHTML=runs.length?`<table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`:'<span class=dim>无 run</span>';
}
function sortBy(i){if(sortCol===i)sortDir=-sortDir;else{sortCol=i;sortDir=1;}render();}

async function detail(id){const r=(RAW.runs||[]).find(x=>x.id===id);if(!r)return;
 if(!r._loaded&&!(r.config&&Object.keys(r.config).length)){await ensureHistory(id);render();}
 const kv=o=>Object.keys(o||{}).sort().map(k=>`<tr><td>${k}</td><td>${typeof o[k]==='number'?fmtNum(o[k]):String(o[k])}</td></tr>`).join('')||'<tr><td class=dim colspan=2>(空)</td></tr>';
 q('modalbox').innerHTML=`<div style=display:flex;align-items:center;gap:10px><span style=color:${COLOR[r.id]}>■</span><b style=font-size:14px>${r.name}</b><span class="dim st-${r.state}">${r.state} · ${r.project}</span><span class=spacer></span><button onclick="q('modal').style.display='none'">关闭</button></div>
  <div class=dim style=margin-top:4px>#${r.id} · runtime ${fmtRt(r.runtime)} · created ${r.created||'—'}</div>
  <div class=sec>summary(终值)</div><table>${kv(r.summary)}</table>
  <div class=sec>config</div><table>${kv(r.config)}</table>`;
 q('modal').style.display='flex';
}
load();setInterval(load,30000);
</script></body></html>"""


routes = [
    Route("/", index),
    Route("/health", health, methods=["GET"]),
    Route("/api/wandb/runs", api_runs, methods=["GET"]),
    Route("/api/wandb/history", api_history, methods=["GET"]),
]

app = Starlette(routes=routes)
