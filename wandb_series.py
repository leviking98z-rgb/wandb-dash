"""wandb_series — 拉 wandb entity 下近期/活跃 run 的 metric 序列, 输出 JSON。

用【系统 python3】跑(它装了 wandb 0.28.0; console venv 没装, 保持干净)。WANDB_API_KEY
从环境取(wandbdash 运行时从 .clusters/.tools/mon_wandb.sh 抽, 不把密钥写进本仓库)。
wandbdash server shell 出本脚本 + TTL 缓存 + 前端画真曲线。

用法: WANDB_API_KEY=... /usr/bin/python3 wandb_series.py [--entity E --keys reward,loss
       --since-hours 72 --samples 200 --max-runs 8]
"""
import argparse
import datetime as dt
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="leviking98z-zhejiang-university")
    ap.add_argument("--keys", default="reward,loss")
    ap.add_argument("--since-hours", type=float, default=336.0)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--max-runs", type=int, default=8)
    ap.add_argument("--max-metrics", type=int, default=6)
    ap.add_argument("--per-project", type=int, default=6, help="每 project 看最新几个 run")
    a = ap.parse_args()
    subs = [k.strip() for k in a.keys.split(",") if k.strip()]

    out = {"entity": a.entity, "ts": time.time(), "keys": subs, "runs": [], "error": None}
    try:
        import wandb
    except Exception as e:
        out["error"] = f"wandb import 失败: {e}"
        print(json.dumps(out)); return

    try:
        api = wandb.Api(timeout=45)
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = a.since_hours * 3600

        def age_of(r):
            at = getattr(r, "_attrs", {}) or {}
            ts = (at.get("updatedAt") or at.get("heartbeatAt") or at.get("createdAt")
                  or getattr(r, "created_at", None))
            if ts is None:
                return None
            try:
                if isinstance(ts, (int, float)):        # epoch 秒
                    return now.timestamp() - float(ts)
                return (now - dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))).total_seconds()
            except Exception:
                return None

        cand = []
        for p in api.projects(a.entity):
            # 每 project 只看最新几个(order 新→旧, per_page 早停), 防遍历几百个 run
            try:
                runs = api.runs(f"{a.entity}/{p.name}", order="-created_at", per_page=a.per_project)
            except Exception:
                continue
            for i, r in enumerate(runs):
                if i >= a.per_project:
                    break
                cand.append((age_of(r), r, p.name))
        # running 优先; 再按最新(age 小)优先; 无 age 排最后
        cand.sort(key=lambda x: (x[1].state != "running",
                                 x[0] if x[0] is not None else 1e18))
        # 优先窗口内的; 若窗口内不足 max_runs, 用最新的补足(没在跑也有曲线看)
        within = [c for c in cand if c[0] is not None and c[0] <= cutoff]
        chosen = within[:a.max_runs] or cand[:a.max_runs]
        for age, r, proj in chosen[:a.max_runs]:
            mkeys = [k for k in r.summary.keys()
                     if not k.startswith("_") and any(s in k for s in subs)][:a.max_metrics]
            metrics = {}
            for k in mkeys:
                try:
                    pts = [[row.get("_step"), row.get(k), row.get("_timestamp")]
                           for row in r.history(samples=a.samples, keys=[k, "_timestamp"], pandas=False)
                           if isinstance(row.get(k), (int, float))]
                    if len(pts) >= 2:
                        metrics[k] = pts
                except Exception:
                    pass
            # config(仅标量, 截断) + summary(各 metric 终值) —— 供前端 run 详情/表格
            cfg = {}
            try:
                for ck, cv in dict(r.config).items():
                    if ck.startswith("_") or not isinstance(cv, (int, float, str, bool)):
                        continue
                    cfg[ck] = cv
                    if len(cfg) >= 40:
                        break
            except Exception:
                pass
            summ = {}
            try:
                for sk in mkeys:
                    sv = r.summary.get(sk)
                    if isinstance(sv, (int, float)):
                        summ[sk] = sv
            except Exception:
                pass
            rt = r.summary.get("_runtime")
            out["runs"].append({
                "name": r.name, "id": r.id, "project": proj, "state": r.state,
                "age_sec": int(age) if age and age < 1e17 else None,
                "runtime": int(rt) if isinstance(rt, (int, float)) else None,
                "created": str((getattr(r, "_attrs", {}) or {}).get("createdAt") or ""),
                "metrics": metrics, "config": cfg, "summary": summ,
            })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
