"""wandbdash — wandb 训练曲线 dashboard(先选 project → 侧栏选 run → 右主区画曲线)。

srv-wandb:8097。后台 syncer 线程主动把 wandb 下载进本地快照(.cache/, 原子落盘),
HTTP 请求只读快照 → 恒定秒开、重启即热、不内联等 wandb。syncer 每 ~45s 一圈:
在跑的项目每圈刷、已结束项目每 30min 刷一次(数据不变则跳过, 省 API); 请求缺失/硬
过期(>1h, 兜底 syncer 挂了)才内联拉一次。数据仍由 wandb_series.py(本仓 venv)产出:
  --list-projects            列 project(每个探一个最新 run)
  --project X                该 project 全部 run 的元数据 + 前 N 个的曲线
  --run-project P --run-id I 单 run 的曲线+config+summary(按需 + 落盘)
  --set-group ...            设某 run 的 group 并 r.update() 写回 online wandb
端点: /api/wandb/{projects,runs?project=X,history,setgroup(POST)} + /health(含 sync 状态)。
前端: 顶部下拉先选项目; 侧栏可经左侧竖条折叠; 每图带可点图例; 无表格。
  曲线区分 18 色 × 5 线型(实/虚/点/点划)=90 种; group 作为标签显示, 详情弹框可
  编辑 group 并同步 online, wandb tags 只读显示。
  看数值: 桌面悬停; 手机【长按/拖动曲线】显示(吸附到真实数据点, 不看插值中间值)。
  缩放: 桌面拖框选; 手机【单点定范围端点→再点一处放大, 同处再点复位】; 双击复位。
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
_SERIES = str(_REPO / "wandb_series.py")
_MON_WANDB = os.environ.get("MON_WANDB_SH", "/root/shared/.clusters/.tools/mon_wandb.sh")
_CACHE = _REPO / ".cache"        # 本地持久快照(gitignore); 后台 syncer 主动下载, 请求只读它

_TTL = 60                        # (保留)内存新鲜阈值
_SYNC_INTERVAL = float(os.environ.get("WANDB_SYNC_INTERVAL", "45"))   # 后台循环间隔(s)
_IDLE_TTL = float(os.environ.get("WANDB_IDLE_TTL", "1800"))          # 已结束项目重刷间隔(s)
_STALE_HARD = 3600.0             # 内存超这么旧才兜底内联重拉(防 syncer 挂了永远陈旧)

_proj_cache: dict = {"ts": 0.0, "data": None}   # project 列表快照
_runs_cache: dict = {}   # project -> (ts, data); 每个 project 的 run 列表快照
_hist_cache: dict = {}   # (project,id) -> (ts, run_dict); 单 run 曲线快照
_lock = threading.Lock()
_syncer_started = False
_sync_stat: dict = {"last": 0.0, "projects": 0, "runs": 0, "err": None}


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "")


def _read_json(name: str):
    try:
        p = _CACHE / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _write_json(name: str, obj) -> None:
    """原子落盘(写临时文件再 rename)。"""
    try:
        _CACHE.mkdir(exist_ok=True)
        p = _CACHE / name
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        pass


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


def _series(*args, timeout=120) -> dict:
    env = {**os.environ, "WANDB_API_KEY": _api_key(), "WANDB_SILENT": "true"}
    p = subprocess.run([sys.executable, _SERIES, *args],
                       capture_output=True, text=True, timeout=timeout, env=env)
    if p.stdout.strip():
        return json.loads(p.stdout)
    return {"error": (p.stderr or "空输出")[:300]}


# ---- 快照写入(内存 + 磁盘); 无锁: GIL 下单个赋值原子, 且几乎只有 syncer 单写者 ----
def _set_projects(data: dict) -> None:
    ts = time.time()
    _proj_cache["ts"], _proj_cache["data"] = ts, data
    _write_json("projects.json", {"ts": ts, "data": data})


def _set_runs(project: str, data: dict) -> None:
    ts = time.time()
    _runs_cache[project] = (ts, data)
    _write_json(f"runs__{_safe(project)}.json", {"ts": ts, "project": project, "data": data})


def _set_hist(key, run: dict) -> None:
    ts = time.time()
    _hist_cache[key] = (ts, run)
    _write_json(f"hist__{_safe(key[0])}__{_safe(key[1])}.json",
                {"ts": ts, "key": list(key), "run": run})


def _apply_group(project: str, rid: str, group: str) -> None:
    """写回成功后乐观更新本地快照(wandb 读回有几秒延迟, 别 refetch), 并落盘。"""
    hit = _runs_cache.get(project)
    if hit:
        for r in hit[1].get("runs", []):
            if r.get("id") == rid:
                r["group"] = group
        _write_json(f"runs__{_safe(project)}.json", {"ts": hit[0], "project": project, "data": hit[1]})
    key = (project, rid)
    if key in _hist_cache:
        ts, run = _hist_cache[key]
        run["group"] = group
        _set_hist(key, run)


def _seed_from_disk() -> None:
    """启动时把上次落盘的快照读回内存 → 重启即热, 不用等 wandb。"""
    pj = _read_json("projects.json")
    if isinstance(pj, dict) and "data" in pj:
        _proj_cache["ts"], _proj_cache["data"] = pj.get("ts", 0.0), pj["data"]
    if not _CACHE.is_dir():
        return
    try:
        for f in _CACHE.glob("runs__*.json"):
            d = _read_json(f.name)
            if isinstance(d, dict) and d.get("project") and "data" in d:
                _runs_cache[d["project"]] = (d.get("ts", 0.0), d["data"])
        for f in _CACHE.glob("hist__*.json"):
            d = _read_json(f.name)
            if isinstance(d, dict) and d.get("key") and "run" in d:
                _hist_cache[tuple(d["key"])] = (d.get("ts", 0.0), d["run"])
    except Exception:
        pass


# ---- 请求路径: 只读快照(几乎总命中); 缺失或硬过期才兜底内联拉一次 ----
def _fetch_projects() -> dict:
    now = time.time()
    if _proj_cache["data"] is not None and now - _proj_cache["ts"] < _STALE_HARD:
        return {**_proj_cache["data"], "cached": round(now - _proj_cache["ts"], 1)}
    with _lock:
        now = time.time()
        if _proj_cache["data"] is not None and now - _proj_cache["ts"] < _STALE_HARD:
            return {**_proj_cache["data"], "cached": round(now - _proj_cache["ts"], 1)}
        try:
            d = _series("--list-projects", timeout=90)
        except Exception as e:
            d = {"error": f"{type(e).__name__}: {e}", "projects": []}
        if d.get("projects") is not None:
            _set_projects(d)
        return {**d, "cached": 0.0}


def _fetch_runs(project: str) -> dict:
    now = time.time()
    hit = _runs_cache.get(project)
    if hit and now - hit[0] < _STALE_HARD:
        return {**hit[1], "cached": round(now - hit[0], 1)}
    with _lock:
        hit = _runs_cache.get(project)
        now = time.time()
        if hit and now - hit[0] < _STALE_HARD:
            return {**hit[1], "cached": round(now - hit[0], 1)}
        try:
            d = _series("--project", project, timeout=120)
        except subprocess.TimeoutExpired:
            d = {"error": "wandb 拉取超时(>120s)", "runs": []}
        except Exception as e:
            d = {"error": f"{type(e).__name__}: {e}", "runs": []}
        if d.get("runs") is not None:
            _set_runs(project, d)
        return {**d, "cached": 0.0}


# ---- 后台 syncer: 主动把 wandb 下载进本地快照 ----
def _sync_once() -> int:
    pj = _series("--list-projects", timeout=90)
    projs = pj.get("projects") or []
    if projs or pj.get("projects") is not None:
        _set_projects(pj)
    now = time.time()
    n = 0
    for p in projs:                     # 已按"在跑的在前"排序 → 活跃项目先刷
        name = p.get("name")
        if not name:
            continue
        hit = _runs_cache.get(name)
        last = hit[0] if hit else None
        # 从没拉过 / 在跑(数据在变) / 已结束但超过重刷间隔 → 才拉; 否则跳过(省 API)
        due = last is None or p.get("running") or (now - last > _IDLE_TTL)
        if not due:
            continue
        d = _series("--project", name, "--max-history", "24", timeout=120)
        if d.get("runs") is not None:
            _set_runs(name, d)
            n += 1
    _sync_stat.update(last=time.time(), projects=len(projs), runs=n)
    return n


def _syncer() -> None:
    while True:
        try:
            _sync_once()
            _sync_stat["err"] = None
        except Exception as e:
            _sync_stat["err"] = f"{type(e).__name__}: {e}"
        time.sleep(_SYNC_INTERVAL)


def _start_syncer() -> None:
    global _syncer_started
    if _syncer_started:
        return
    _syncer_started = True
    threading.Thread(target=_syncer, name="wandb-syncer", daemon=True).start()


async def health(request):
    now = time.time()
    return JSONResponse({
        "status": "ok", "service": "wandb", "version": _VERSION,
        "snapshot": {
            "have_projects": _proj_cache["data"] is not None,
            "cached_projects": len(_runs_cache),
            "cached_runs_curves": len(_hist_cache),
            "last_sync_age": round(now - _sync_stat["last"], 1) if _sync_stat["last"] else None,
            "sync_err": _sync_stat["err"],
        },
    })


async def api_projects(request):
    return JSONResponse(_fetch_projects())


async def api_runs(request):
    proj = request.query_params.get("project", "")
    if not proj:
        return JSONResponse({"error": "缺 project", "runs": []}, status_code=400)
    return JSONResponse(_fetch_runs(proj))


async def api_history(request):
    """按需拉单个 run 的曲线 + config + summary(列表模式为省时不带这些)。"""
    proj = request.query_params.get("project", "")
    rid = request.query_params.get("id", "")
    if not proj or not rid:
        return JSONResponse({"error": "缺 project/id"}, status_code=400)
    key = (proj, rid)
    now = time.time()
    hit = _hist_cache.get(key)
    if hit and now - hit[0] < _STALE_HARD:   # 快照够用直接返回(已结束 run 的曲线不变)
        return JSONResponse({**hit[1], "cached": round(now - hit[0], 1)})
    try:
        d = _series("--run-project", proj, "--run-id", rid, timeout=60)
    except Exception as e:
        d = {"error": f"{type(e).__name__}: {e}"}
    run = d.get("run") or {"metrics": {}, "config": {}, "summary": {}, "error": d.get("error")}
    _set_hist(key, run)
    return JSONResponse({**run, "cached": 0.0})


async def api_setgroup(request):
    """设置某 run 的 group 并同步到 online wandb; 成功后乐观更新本地快照。"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    proj = (body.get("project") or "").strip()
    rid = (body.get("id") or "").strip()
    grp = (body.get("group") or "").strip()
    if not proj or not rid:
        return JSONResponse({"error": "缺 project/id"}, status_code=400)
    try:
        d = _series("--set-group", "--run-project", proj, "--run-id", rid, "--group", grp, timeout=60)
    except Exception as e:
        d = {"error": f"{type(e).__name__}: {e}"}
    if d.get("ok"):
        _apply_group(proj, rid, d.get("group", ""))
    return JSONResponse(d)


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
#wrap{display:flex;gap:0;align-items:flex-start}
#side{width:290px;flex:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;position:sticky;top:8px;max-height:calc(100vh - 20px);overflow:auto;transition:width .2s,padding .2s,opacity .15s}
body.collapsed #side{width:0;padding-left:0;padding-right:0;border-width:0;opacity:0;overflow:hidden;pointer-events:none}
#grip{flex:none;width:20px;align-self:stretch;min-height:calc(100vh - 20px);position:sticky;top:8px;margin:0 6px;background:var(--card);border:1px solid var(--line);border-radius:6px;cursor:pointer;display:flex;align-items:center;justify-content:center;color:var(--dim);user-select:none}
#grip:hover{border-color:#58a6ff;color:#58a6ff;background:var(--sel)}
#grip span{writing-mode:vertical-rl;font-size:12px;letter-spacing:3px}
#main{flex:1;min-width:0}
#proj{width:100%;background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;margin-bottom:6px;font:inherit}
#filter{width:100%;background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;margin-bottom:6px;font:inherit}
.run{display:flex;align-items:flex-start;gap:6px;padding:5px 4px;border-radius:6px;cursor:default}
.run:hover{background:var(--sel)}
.run .nm{color:#58a6ff;cursor:pointer;word-break:break-all}
.run .props{color:var(--dim);font-size:10.5px}
.sw{flex:none;margin-top:2px;vertical-align:middle}
.tag{display:inline-block;background:#1f6feb26;color:#79c0ff;border:1px solid #1f6feb55;border-radius:4px;padding:0 5px;font-size:9.5px;vertical-align:middle}
.st-running{color:#3fb950}.st-crashed,.st-failed,.st-killed{color:#f85149}.st-finished{color:#8b949e}
#pager{display:flex;align-items:center;gap:8px;justify-content:center;margin-top:8px;font-size:11px}
#pager button{padding:1px 8px}
#ctl{display:flex;align-items:center;gap:16px;flex-wrap:wrap;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin-bottom:10px;font-size:12px}
#ctl label{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
#charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.card h3{margin:2px 6px 4px;font-size:12px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:3px 12px;margin:6px 4px 0;font-size:10.5px}
.legend span{display:inline-flex;align-items:center;gap:5px;cursor:pointer;color:var(--dim);max-width:210px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.legend span:hover{color:var(--fg)}
.legend i{width:12px;height:3px;border-radius:2px;flex:none}
svg text{fill:var(--dim);font-size:10px}
#tip{position:fixed;pointer-events:none;background:#1f2733;border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:11px;display:none;z-index:9;max-width:260px;box-shadow:0 2px 8px #0008}
#tip b{color:var(--fg)}
details{margin-top:12px}summary{cursor:pointer;color:#58a6ff;font-weight:600}
#sideback{display:none;position:fixed;inset:0;background:#0008;z-index:29}
table{border-collapse:collapse;width:100%;font-size:11px;margin-top:6px}
th,td{border:1px solid var(--line);padding:3px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.name{color:#58a6ff;cursor:pointer}
#modal{position:fixed;inset:0;background:#000a;display:none;align-items:center;justify-content:center;z-index:20}
#modalbox{background:var(--card);border:1px solid var(--line);border-radius:10px;max-width:90vw;max-height:80vh;overflow:auto;padding:16px;min-width:300px}
.sec{margin:14px 0 4px;font-size:13px;font-weight:600;color:#58a6ff}
@media(max-width:640px){
 #side{position:fixed;top:0;left:0;height:100vh;width:84vw;max-width:320px;max-height:none;z-index:30;border-radius:0;opacity:1;transition:transform .2s}
 body:not(.collapsed) #side{transform:none;box-shadow:2px 0 16px #000a;pointer-events:auto}
 body.collapsed #side{transform:translateX(-102%);width:84vw;padding:8px;border-width:0;opacity:1}
 body:not(.collapsed) #sideback{display:block}
 #grip{position:fixed;left:0;top:0;height:100vh;width:20px;z-index:31;border-radius:0;margin:0}
}
</style></head><body>
<header><h1>📈 wandb</h1><span class=dim id=sub></span><span class=spacer></span>
<span class=dim id=upd></span><button onclick=reload()>刷新</button></header>
<div id=sideback onclick=collapseSide()></div>
<div id=wrap>
 <aside id=side>
  <select id=proj onchange="selectProject(this.value)"></select>
  <input id=filter placeholder="过滤 run(名/状态)…" oninput="page=0;render()">
  <div id=runlist></div>
  <div id=pager></div>
 </aside>
 <div id=grip onclick=toggleSide()><span>运行 ▸</span></div>
 <main id=main>
  <div id=ctl>
   <label>平滑 <input type=range id=sm min=0 max=0.95 step=0.05 value=0 oninput=apply()><span id=smv>0.00</span></label>
   <label><input type=checkbox id=logy onchange=apply()> 对数 Y</label>
   <label>X:
     <label><input type=radio name=xm id=xstep checked onchange=apply()> step</label>
     <label><input type=radio name=xm id=xtime onchange=apply()> 时间</label></label>
   <button onclick=resetZoom()>复位缩放</button>
   <span class=dim id=cthint>拖框放大 · 双击复位 · 悬停看值 · 点图例/勾选切换 run</span>
  </div>
  <div id=charts>加载中…</div>
 </main>
</div>
<div id=tip></div>
<div id=modal onclick="if(event.target.id=='modal')this.style.display='none'"><div id=modalbox></div></div>
<script>
const PAL=['#58a6ff','#3fb950','#f2cc60','#f85149','#bc8cff','#39c5cf','#ff7b72','#e3b341','#ff9bce','#7ee787','#79c0ff','#d2a8ff','#ffa657','#56d4dd','#f778ba','#a5d6ff','#ffdf5d','#7ce38b'];
const DASHES=['','6,3','1,3','9,4,2,4','3,3'];   // 实线/长虚/点线/点划/短虚 —— 颜色用完再叠线型, 18×5=90 种
let RAW={runs:[]}; const hidden=new Set(); const zoom={}; const COLOR={}; const DASH={}; let inited=false;
let PROJECTS=[]; let curProj=null;
let smooth=0,logY=false,xmode='step'; let page=0; const PS=10;
const q=id=>document.getElementById(id), TIP=()=>q('tip');
const nMetrics=r=>Object.keys(r&&r.metrics||{}).length;
const isTouch=ev=>!!(ev&&ev.pointerType&&ev.pointerType!=='mouse');   // 触屏/触控笔
const TOUCH=('ontouchstart' in window)||navigator.maxTouchPoints>0;
function fmtNum(v){if(v==null||isNaN(v))return '—';const a=Math.abs(v);if(a!==0&&(a<1e-3||a>=1e5))return v.toExponential(2);return (Math.round(v*1e4)/1e4).toString();}
function fmtX(x){if(xmode==='time'){const d=new Date(x);return isNaN(d)?'-':d.toLocaleTimeString('zh',{hour12:false});}return Math.round(x).toString();}
function fmtRt(s){if(s==null)return '—';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);if(h)return h+'h'+m+'m';if(m)return m+'m'+(s%60)+'s';return s+'s';}
function getX(p){return xmode==='time'?(p[2]?p[2]*1000:NaN):p[0];}
function ema(pts,a){if(!(a>0))return pts.map(p=>[p.x,p.y]);let s=null,o=[];for(const p of pts){s=s==null?p.y:a*s+(1-a)*p.y;o.push([p.x,s]);}return o;}

function apply(){smooth=parseFloat(q('sm').value)||0;q('smv').textContent=smooth.toFixed(2);logY=q('logy').checked;xmode=q('xstep').checked?'step':'time';render();}
function resetZoom(){for(const k in zoom)delete zoom[k];render();}
async function toggleRun(id){if(hidden.has(id)){hidden.delete(id);await ensureHistory(id);}else{hidden.add(id);}render();}
function updateGrip(){const c=document.body.classList.contains('collapsed');q('grip').innerHTML='<span>'+(c?'运行 ▸':'◂ 收起')+'</span>';}
function toggleSide(){document.body.classList.toggle('collapsed');updateGrip();}
function collapseSide(){document.body.classList.add('collapsed');updateGrip();}

async function boot(){
 try{const r=await fetch('/api/wandb/projects');const d=await r.json();
   PROJECTS=d.projects||[];
   q('sub').textContent=(d.entity||'')+(d.error?(' · ⚠'+d.error):'');
   q('proj').innerHTML=PROJECTS.map(p=>`<option value="${p.name}">${p.running?'● ':''}${p.name}</option>`).join('');
   const def=(PROJECTS.find(p=>p.running)||PROJECTS[0]||{}).name;
   if(def){q('proj').value=def;await selectProject(def);}else q('charts').textContent='无 project';
 }catch(e){q('charts').textContent='加载 project 失败: '+e;}
}
async function selectProject(name){
 curProj=name;inited=false;page=0;hidden.clear();RAW={runs:[]};
 q('charts').textContent='加载 '+name+' …';q('runlist').innerHTML='';q('pager').innerHTML='';
 await loadRuns();
}
function reload(){loadRuns();}   // 刷新键 & 30s 轮询: 重拉当前 project

async function loadRuns(){if(!curProj)return;try{
 const r=await fetch('/api/wandb/runs?project='+encodeURIComponent(curProj));const nw=await r.json();
 // 合并: 保留已按需拉到的曲线/config, 避免 30s 轮询把它们冲掉
 const prev={};(RAW.runs||[]).forEach(x=>prev[x.id]=x);
 (nw.runs||[]).forEach(x=>{const o=prev[x.id];if(o){if(!nMetrics(x)&&nMetrics(o))x.metrics=o.metrics;if(o._loaded){x._loaded=true;if(o.config&&Object.keys(o.config).length)x.config=o.config;if(o.summary&&Object.keys(o.summary).length&&!Object.keys(x.summary||{}).length)x.summary=o.summary;}}});
 RAW=nw;const runs=RAW.runs||[];
 if(RAW.error)q('sub').textContent=(RAW.entity||'')+' · ⚠'+RAW.error;
 const nc=runs.filter(r=>nMetrics(r)).length;
 q('upd').textContent='更新 '+(RAW.cached?RAW.cached+'s前':'刚刚')+' · '+curProj+' · '+runs.length+' run · '+nc+' 有曲线';
 runs.forEach((r,i)=>{if(!COLOR[r.id]){COLOR[r.id]=PAL[i%PAL.length];DASH[r.id]=DASHES[Math.floor(i/PAL.length)%DASHES.length];}});
 if(!inited){inited=true;
   // 默认只画在跑的(没有则画前几个有曲线的), 其余 run 都在侧栏列表里可勾选
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
 // 侧栏: 过滤(选定 project 内) + 分页
 const f=(q('filter').value||'').toLowerCase();
 const filt=runs.filter(r=>!f||(r.name+' '+r.state+' '+(r.group||'')+' '+((r.tags||[]).join(' '))).toLowerCase().includes(f));
 const pages=Math.max(1,Math.ceil(filt.length/PS)); if(page>=pages)page=pages-1; if(page<0)page=0;
 const paged=filt.slice(page*PS,page*PS+PS);
 q('runlist').innerHTML=paged.map(r=>{
   const off=hidden.has(r.id);
   const mark=r._loading?' <span class=dim>⏳</span>':(nMetrics(r)?' <span title="有曲线">📈</span>':'');
   const sw=`<svg class=sw width=16 height=10><line x1=0 y1=5 x2=16 y2=5 stroke="${COLOR[r.id]}" stroke-width=2.5 stroke-dasharray="${DASH[r.id]||''}"></line></svg>`;
   const grp=r.group?` <span class=tag>${r.group}</span>`:'';
   return `<div class=run><input type=checkbox ${off?'':'checked'} onchange="toggleRun('${r.id}')">${sw}<div><span class=nm onclick="detail('${r.id}')">${r.name}</span>${mark}${grp}<div class=props><span class="st-${r.state}">${r.state}</span> · ⏱${fmtRt(r.runtime)} · #${(r.id||'').slice(0,8)}</div></div></div>`;
 }).join('')||'<div class=dim style=padding:8px>无匹配 run</div>';
 q('pager').innerHTML=filt.length?`<button onclick="page--;render()" ${page<=0?'disabled':''}>‹</button><span class=dim>${page*PS+1}–${Math.min((page+1)*PS,filt.length)} / ${filt.length}${f?' (筛后)':''}</span><button onclick="page++;render()" ${page>=pages-1?'disabled':''}>›</button>`:'';
 // 主区: 图表(画所有勾选=未 hidden 的 run, 不受分页/过滤影响)
 const vis=runs.filter(r=>!hidden.has(r.id));
 const metrics={};vis.forEach(r=>{for(const m in (r.metrics||{}))(metrics[m]=metrics[m]||[]).push(r);});
 const keys=Object.keys(metrics).sort();
 const cont=q('charts');cont.innerHTML='';
 if(!keys.length)cont.innerHTML='<span class=dim>暂无曲线 —— 侧栏勾选 run 即按需加载其曲线(默认只画在跑的)</span>';
 keys.forEach(m=>{const c=document.createElement('div');c.className='card';cont.appendChild(c);drawChart(c,m,metrics[m]);});
}

function drawChart(card,metric,runsForMetric){
 const W=560,H=210,pl=54,pr=12,pt=10,pb=24;
 let series=runsForMetric.map(r=>{
   let pts=(r.metrics[metric]||[]).map(p=>({x:getX(p),y:p[1]})).filter(p=>isFinite(p.x)&&isFinite(p.y)).sort((a,b)=>a.x-b.x);
   const z=zoom[metric];if(z)pts=pts.filter(p=>p.x>=z[0]&&p.x<=z[1]);
   return {id:r.id,name:r.name,color:COLOR[r.id],dash:DASH[r.id]||'',pts:ema(pts,smooth)};
 }).filter(s=>s.pts.length>=1);
 const legend='<div class=legend>'+series.map(s=>`<span onclick="toggleRun('${s.id}')" title="点击隐藏 ${s.name}"><svg class=sw width=20 height=8><line x1=0 y1=4 x2=20 y2=4 stroke="${s.color}" stroke-width=2 stroke-dasharray="${s.dash}"></line></svg>${s.name}</span>`).join('')+'</div>';
 let xs=[],ys=[];series.forEach(s=>s.pts.forEach(p=>{xs.push(p[0]);ys.push(p[1]);}));
 if(xs.length<2){card.innerHTML=`<h3>${metric}</h3><div class=dim style=padding:16px>无点</div>${legend}`;return;}
 const x0=Math.min(...xs),x1=Math.max(...xs);
 let yv=ys.slice();if(logY){const pos=yv.filter(v=>v>0);yv=pos.length?pos.map(v=>Math.log10(v)):yv;}
 let y0=Math.min(...yv),y1=Math.max(...yv);if(y0===y1){y0-=1;y1+=1;}
 const xr=(x1-x0)||1,yr=(y1-y0)||1;
 const X=v=>pl+(v-x0)/xr*(W-pl-pr);
 const YV=v=>logY?(v>0?Math.log10(v):y0):v;
 const Y=v=>H-pb-(YV(v)-y0)/yr*(H-pt-pb);
 // 只在【真实数据点】的 x 上取值(不看插值中间值, 同 wandb)
 const gridX=[...new Set(series.flatMap(s=>s.pts.map(p=>p[0])))].sort((a,b)=>a-b);
 let grid='';for(let i=0;i<=3;i++){const t=y0+yr*i/3;const py=H-pb-(t-y0)/yr*(H-pt-pb);grid+=`<line x1="${pl}" y1="${py.toFixed(1)}" x2="${W-pr}" y2="${py.toFixed(1)}" stroke="#21262d"></line><text x="6" y="${(py+3).toFixed(1)}">${fmtNum(logY?Math.pow(10,t):t)}</text>`;}
 const lines=series.map(s=>`<polyline fill="none" stroke="${s.color}" stroke-width="1.5" stroke-dasharray="${s.dash}" points="${s.pts.map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ')}"></polyline>`).join('');
 const xl=`<text x="${pl}" y="${H-6}">${fmtX(x0)}</text><text x="${W-pr}" y="${H-6}" text-anchor="end">${fmtX(x1)}</text>`;
 card.innerHTML=`<h3>${metric}${zoom[metric]?' <span class=dim style=font-weight:400>· 已缩放(双击复位)</span>':''}</h3>
  <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block;touch-action:none">${grid}${lines}${xl}
  <g class="dots"></g>
  <line class="cross" y1="${pt}" y2="${H-pb}" stroke="#8b949e" stroke-dasharray="3,3" style="display:none"></line>
  <line class="marker" y1="${pt}" y2="${H-pb}" stroke="#58a6ff" stroke-width="1.5" style="display:none"></line>
  <rect class="band" y="${pt}" height="${H-pt-pb}" fill="#58a6ff33" style="display:none"></rect>
  <rect class="ov" x="${pl}" y="${pt}" width="${W-pl-pr}" height="${H-pt-pb}" fill="transparent" style="cursor:crosshair"></rect></svg>${legend}`;
 const svg=card.querySelector('svg'),cross=svg.querySelector('.cross'),band=svg.querySelector('.band'),ov=svg.querySelector('.ov'),dots=svg.querySelector('.dots'),marker=svg.querySelector('.marker');
 const evX=ev=>{const r=svg.getBoundingClientRect();const sx=(ev.clientX-r.left)/r.width*W;return{sx,dx:x0+(sx-pl)/(W-pl-pr)*xr};};
 const hideTip=()=>{cross.style.display='none';TIP().style.display='none';dots.innerHTML='';};
 const inspect=ev=>{const {dx}=evX(ev);if(!gridX.length)return;
   let snap=gridX[0],bd=1e18;for(const gx of gridX){const d=Math.abs(gx-dx);if(d<bd){bd=d;snap=gx;}}   // 吸附到最近真实数据点
   const px=X(snap);cross.setAttribute('x1',px);cross.setAttribute('x2',px);cross.style.display='';
   let html=`<b>${metric}</b> · ${xmode==='time'?'时间':'step'} ${fmtX(snap)}<br>`,dh='';
   series.forEach(s=>{let best=null,bb=1e18;for(const p of s.pts){const d=Math.abs(p[0]-snap);if(d<bb){bb=d;best=p;}}
     if(best){html+=`<span style=color:${s.color}>■</span> ${s.name}: <b>${fmtNum(best[1])}</b><br>`;
       dh+=`<circle cx="${X(best[0]).toFixed(1)}" cy="${Y(best[1]).toFixed(1)}" r="3.2" fill="${s.color}" stroke="#0d1117" stroke-width="0.8"></circle>`;}});
   dots.innerHTML=dh;
   const t=TIP();t.innerHTML=html;t.style.display='block';
   const left=Math.max(8,Math.min(ev.clientX+12,innerWidth-t.offsetWidth-8));
   let top=isTouch(ev)?(ev.clientY-t.offsetHeight-22):(ev.clientY+12);   // 触屏: tooltip 放手指【上方】不被挡
   if(top<6)top=ev.clientY+22;
   t.style.left=left+'px';t.style.top=top+'px';};
 // ------- 交互 -------
 // 鼠标(不变): 悬停看值 · 拖框放大 · 双击复位
 // 触屏(按用户方案): 长按/拖动=看数值 · 单点=定范围端点(出蓝线) · 再点另一处=框选放大 · 同处再点=复位
 let drag=null;                     // 鼠标框选
 let rangeA=null;                   // 触屏第一个范围端点(data-x)
 let downCX=0,downT=0,moved=false,lpActive=false,lpTimer=0;   // 触屏手势判定
 const clearRange=()=>{rangeA=null;marker.style.display='none';};
 ov.addEventListener('pointermove',ev=>{
   if(drag!=null){const sx=evX(ev).sx;const a=X(drag);band.setAttribute('x',Math.min(a,sx));band.setAttribute('width',Math.abs(sx-a));band.style.display='';cross.style.display='none';TIP().style.display='none';dots.innerHTML='';return;}
   if(isTouch(ev)){
     if(!lpActive&&Math.abs(ev.clientX-downCX)>8){moved=true;lpActive=true;if(lpTimer){clearTimeout(lpTimer);lpTimer=0;}}  // 拖动即看数值
     if(lpActive)inspect(ev);
     return;}
   inspect(ev);});   // 鼠标 hover
 ov.addEventListener('pointerdown',ev=>{ov.setPointerCapture(ev.pointerId);
   if(!isTouch(ev)){drag=evX(ev).dx;return;}                                  // 鼠标: 框选
   downCX=ev.clientX;downT=Date.now();moved=false;lpActive=false;
   lpTimer=setTimeout(()=>{lpActive=true;inspect(ev);},350);});               // 长按: 显数值
 ov.addEventListener('pointerup',ev=>{
   if(drag!=null){const dx=evX(ev).dx;const a=Math.min(drag,dx),b=Math.max(drag,dx);drag=null;band.style.display='none';if(b-a>xr*0.02){zoom[metric]=[a,b];render();}return;}
   if(!isTouch(ev))return;
   if(lpTimer){clearTimeout(lpTimer);lpTimer=0;}
   if(!moved&&!lpActive&&Date.now()-downT<350){                              // === 一次"单点" ===
     const dx=evX(ev).dx;
     if(rangeA==null){rangeA=dx;marker.setAttribute('x1',X(dx));marker.setAttribute('x2',X(dx));marker.style.display='';hideTip();}  // 定第一个端点(蓝线)
     else{const a=Math.min(rangeA,dx),b=Math.max(rangeA,dx);clearRange();
       if(b-a<xr*0.03){delete zoom[metric];}   // 同处再点 → 复位
       else{zoom[metric]=[a,b];}               // 另一处 → 框选放大
       render();}
     return;}
   lpActive=false;});   // 长按/拖动看数值 抬手: 保留 tooltip(轻点别处消失)
 ov.addEventListener('pointercancel',ev=>{if(lpTimer){clearTimeout(lpTimer);lpTimer=0;}lpActive=false;});
 ov.addEventListener('pointerleave',ev=>{if(!isTouch(ev))hideTip();});
 svg.addEventListener('dblclick',()=>{if(zoom[metric]){delete zoom[metric];render();}});
}

async function detail(id){const r=(RAW.runs||[]).find(x=>x.id===id);if(!r)return;
 if(!r._loaded&&!(r.config&&Object.keys(r.config).length)){await ensureHistory(id);render();}
 const kv=o=>Object.keys(o||{}).sort().map(k=>`<tr><td>${k}</td><td>${typeof o[k]==='number'?fmtNum(o[k]):String(o[k])}</td></tr>`).join('')||'<tr><td class=dim colspan=2>(空)</td></tr>';
 const esc=s=>String(s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
 const tagsHtml=(r.tags&&r.tags.length)?`<div class=sec>tags(wandb)</div><div>${r.tags.map(t=>`<span class=tag>${t}</span>`).join(' ')}</div>`:'';
 q('modalbox').innerHTML=`<div style=display:flex;align-items:center;gap:10px><span style=color:${COLOR[r.id]}>■</span><b style=font-size:14px>${r.name}</b><span class="dim st-${r.state}">${r.state} · ${r.project}</span><span class=spacer></span><button onclick="q('modal').style.display='none'">关闭</button></div>
  <div class=dim style=margin-top:4px>#${r.id} · runtime ${fmtRt(r.runtime)} · created ${r.created||'—'}</div>
  <div class=sec>group(编辑并同步到 online wandb)</div>
  <div style="display:flex;gap:6px;align-items:center"><input id=grpin value="${esc(r.group)}" placeholder="(无 group, 可填写)" style="flex:1;background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font:inherit"><button onclick="saveGroup('${r.id}')">保存并同步</button></div>
  <div id=grpstat class=dim style=margin-top:4px></div>
  ${tagsHtml}
  <div class=sec>summary(终值)</div><table>${kv(r.summary)}</table>
  <div class=sec>config</div><table>${kv(r.config)}</table>`;
 q('modal').style.display='flex';
}
async function saveGroup(id){const r=(RAW.runs||[]).find(x=>x.id===id);if(!r)return;
 const g=(q('grpin').value||'').trim(); const st=q('grpstat');
 st.textContent='保存中… 正在同步到 online wandb';
 try{const res=await fetch('/api/wandb/setgroup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project:r.project,id:r.id,group:g})});
   const d=await res.json();
   if(d.ok){r.group=d.group||'';st.textContent='已保存并同步 ✓  group = '+(d.group||'(空)');render();}
   else st.textContent='失败: '+(d.error||'未知错误');
 }catch(e){st.textContent='失败: '+e;}
}
if(TOUCH){q('cthint').textContent='长按/拖动看数值(吸附数据点) · 单点定范围端点 · 再点一处放大 · 同处再点复位';}
// 触屏: 轻点图表外任意处 → 收起钉住的数值气泡与高亮点
document.addEventListener('pointerdown',ev=>{if(isTouch(ev)&&!(ev.target&&ev.target.classList&&ev.target.classList.contains('ov'))){TIP().style.display='none';document.querySelectorAll('svg .cross').forEach(c=>c.style.display='none');document.querySelectorAll('svg .dots').forEach(g=>g.innerHTML='');}});
if(innerWidth<=640)document.body.classList.add('collapsed');
updateGrip(); boot(); setInterval(()=>{if(curProj)loadRuns();},30000);
</script></body></html>"""


routes = [
    Route("/", index),
    Route("/health", health, methods=["GET"]),
    Route("/api/wandb/projects", api_projects, methods=["GET"]),
    Route("/api/wandb/runs", api_runs, methods=["GET"]),
    Route("/api/wandb/history", api_history, methods=["GET"]),
    Route("/api/wandb/setgroup", api_setgroup, methods=["POST"]),
]

_seed_from_disk()   # 导入即用上次落盘的快照(重启即热)
_start_syncer()     # 后台 syncer 线程(daemon), 主动下载 wandb → 本地快照
app = Starlette(routes=routes)
