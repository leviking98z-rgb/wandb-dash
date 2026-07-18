"""wandb_series — 拉 wandb entity 下的 run 列表 + metric 序列, 输出 JSON。

用【本仓 venv python】跑(装了 wandb; console venv 没装, 保持干净)。WANDB_API_KEY
从环境取(wandbdash 运行时从 .clusters/.tools/mon_wandb.sh 抽, 不把密钥写进本仓库)。
wandbdash server shell 出本脚本 + TTL 缓存 + 前端画真曲线。

两种模式:
  列表模式(默认): 枚举 entity 下【全部】 run 的元数据(便宜, 只读 _attrs, 不碰 r.summary
    —— 后者会对每个 run 惰性 HTTP 拉 summary 文件, 几百个 run 要几分钟), 再【只给最新的
    --max-history 个】并行拉曲线(history 不传 keys, 跳过 summary 访问)。其余 run 也在列表里,
    曲线按需(前端勾选时打 --run-* 单拉)。
  单 run 模式: --run-project P --run-id I → 只拉这一个 run 的曲线(前端按需)。

用法: WANDB_API_KEY=... python wandb_series.py [--entity E --keys reward,loss --samples 200
       --max-history 40 --max-runs 1000]
      python wandb_series.py --run-project P --run-id I
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import time


def _parse_ts(ts):
    """iso 字符串 / epoch 秒 → aware datetime(UTC), 失败 None。"""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc)
        return dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def _clip(v, n=200):
    s = v if isinstance(v, str) else v
    if isinstance(s, str) and len(s) > n:
        return s[:n] + "…"
    return s


def _config_of(r):
    """从 _attrs(便宜, 无网络)取 config, 展平 wandb 的 {value,desc} 包装。"""
    cfg = {}
    try:
        raw = dict(r.config)
    except Exception:
        return cfg
    for ck, cv in raw.items():
        if ck.startswith("_") or not isinstance(cv, (int, float, str, bool)):
            continue
        cfg[ck] = _clip(cv)
        if len(cfg) >= 40:
            break
    return cfg


def _history_metrics(r, subs, samples, max_metrics):
    """拉一个 run 的曲线。不传 keys → 用采样 history 拿全部指标, 按 subs 过滤,
    从而【不访问 r.summary】(那才是慢的根源)。返回 {metric: [[step,val,ts],...]}。"""
    metrics = {}
    try:
        for row in r.history(samples=samples, pandas=False):
            st, ts = row.get("_step"), row.get("_timestamp")
            for k, v in row.items():
                if k.startswith("_") or not isinstance(v, (int, float)):
                    continue
                if not any(s in k for s in subs):
                    continue
                metrics.setdefault(k, []).append([st, v, ts])
    except Exception:
        pass
    metrics = {k: v for k, v in metrics.items() if len(v) >= 2}
    if len(metrics) > max_metrics:
        metrics = {k: metrics[k] for k in sorted(metrics)[:max_metrics]}
    return metrics


def _summary_terminal(metrics):
    """终值(曲线最后一点)当 summary, 免去访问 r.summary。"""
    return {k: pts[-1][1] for k, pts in metrics.items() if pts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="leviking98z-zhejiang-university")
    ap.add_argument("--keys", default="reward,loss")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--max-runs", type=int, default=2000, help="列表上限(元数据, 便宜)")
    ap.add_argument("--max-history", type=int, default=16, help="首屏自动拉曲线的 run 数(贵)")
    ap.add_argument("--max-metrics", type=int, default=6)
    ap.add_argument("--project", default=None, help="只看这一个 project 的 run")
    ap.add_argument("--list-projects", action="store_true", help="只列 project(便宜, 供先选项目)")
    # 单 run 模式(前端按需拉某个 run 的曲线)
    ap.add_argument("--run-project", default=None)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    subs = [k.strip() for k in a.keys.split(",") if k.strip()]

    out = {"entity": a.entity, "ts": time.time(), "keys": subs, "error": None}
    try:
        import wandb
    except Exception as e:
        out["error"] = f"wandb import 失败: {e}"
        print(json.dumps(out)); return

    try:
        api = wandb.Api(timeout=45)

        # ---- 单 run 模式: 拉这一个 run 的曲线 + config(前端勾选/看详情时按需) ----
        # config/summary 都是惰性 HTTP(每 run 一次), 只在单拉时付这个钱, 不进列表枚举。
        if a.run_id and a.run_project:
            r = api.run(f"{a.entity}/{a.run_project}/{a.run_id}")
            metrics = _history_metrics(r, subs, a.samples, a.max_metrics)
            out["run"] = {"id": a.run_id, "project": a.run_project, "metrics": metrics,
                          "summary": _summary_terminal(metrics), "config": _config_of(r)}
            print(json.dumps(out)); return

        now = dt.datetime.now(dt.timezone.utc)

        def attrs(r):
            return getattr(r, "_attrs", {}) or {}

        def age_of(at):
            d = _parse_ts(at.get("updatedAt") or at.get("heartbeatAt") or at.get("createdAt"))
            return (now - d).total_seconds() if d else None

        def runtime_of(at):
            c = _parse_ts(at.get("createdAt"))
            h = _parse_ts(at.get("heartbeatAt") or at.get("updatedAt"))
            if c and h:
                s = (h - c).total_seconds()
                if s >= 0:
                    return int(s)
            return None

        # ---- 列 project 模式: 每个 project 探一个最新 run(并行, 便宜) → 供先选项目 ----
        if a.list_projects:
            names = [p.name for p in api.projects(a.entity)]

            def probe(name):
                info = {"name": name, "running": False, "latest": None, "state": None}
                try:
                    for r in api.runs(f"{a.entity}/{name}", order="-created_at", per_page=1):
                        at = attrs(r)
                        d = _parse_ts(at.get("updatedAt") or at.get("heartbeatAt") or at.get("createdAt"))
                        info.update(running=(r.state == "running"),
                                    latest=(d.timestamp() if d else None), state=r.state)
                        break
                except Exception:
                    pass
                return info

            with ThreadPoolExecutor(max_workers=8) as ex:
                items = list(ex.map(probe, names))
            # 在跑的在前; 再按最新活动新→旧; 空项目排最后
            items.sort(key=lambda x: (not x["running"], -(x["latest"] or -1e18)))
            out["projects"] = items
            print(json.dumps(out)); return

        # ---- 列表模式: 枚举 run 的元数据(只读 _attrs, 便宜)。--project 则只看该项目 ----
        proj_names = [a.project] if a.project else [p.name for p in api.projects(a.entity)]
        cand = []
        for name in proj_names:
            try:
                runs = api.runs(f"{a.entity}/{name}", order="-created_at", per_page=200)
            except Exception:
                continue
            for r in runs:
                cand.append((r, name))
                if len(cand) >= a.max_runs:
                    break
            if len(cand) >= a.max_runs:
                break

        metas, by_id = [], {}
        for r, proj in cand:
            try:
                at = attrs(r)
                age = age_of(at)
                m = {
                    "name": r.name, "id": r.id, "project": proj, "state": r.state,
                    "age_sec": int(age) if age is not None else None,
                    "runtime": runtime_of(at),
                    "created": str(at.get("createdAt") or ""),
                    # config/summary 惰性 HTTP(每 run 一次, 全量拉 394 个要 ~2.5min) →
                    # 列表里留空, 看详情/画曲线时按需单拉。
                    "metrics": {}, "config": {}, "summary": {},
                }
                metas.append(m)
                by_id[r.id] = (r, m)
            except Exception:
                continue

        # running 优先; 再最新(age 小)优先; 无 age 排最后
        metas.sort(key=lambda m: (m["state"] != "running",
                                  m["age_sec"] if m["age_sec"] is not None else 1e18))

        # ---- 首屏: 只给最新/在跑的前 max_history 个并行拉曲线 ----
        targets = [m["id"] for m in metas[:a.max_history]]

        def fetch_history(rid):
            r, m = by_id[rid]
            metrics = _history_metrics(r, subs, a.samples, a.max_metrics)
            return rid, metrics

        with ThreadPoolExecutor(max_workers=8) as ex:
            for rid, metrics in ex.map(fetch_history, targets):
                _, m = by_id[rid]
                m["metrics"] = metrics
                if metrics:
                    m["summary"] = _summary_terminal(metrics)

        out["runs"] = metas
        out["n_total"] = len(metas)
        out["n_history"] = len(targets)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["runs"] = out.get("runs", [])
    print(json.dumps(out))


if __name__ == "__main__":
    main()
