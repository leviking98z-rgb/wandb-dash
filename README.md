# wandb-dash

wandb 训练曲线 dashboard(网页看真 SVG 曲线)。独立 server(srv-wandb:8097),
从 console 拆出——自带 venv(含 wandb),不再依赖系统 python 装 wandb。

- `wandbdash.py`: Starlette app。`/`=页面, `/health`, `/api/wandb/runs`。
  shell 出 `wandb_series.py`(用**本仓 venv 的 python** = `sys.executable`)拉 entity 的
  metric 序列 → TTL 缓存 → 前端画曲线。
- `wandb_series.py`: 拉 wandb entity 近期/活跃 run 的 metric 序列, 输出 JSON。
- WANDB_API_KEY 运行时从 `.clusters/.tools/mon_wandb.sh` 抽(不写进本仓)。

## 部署(srv)
servers.conf 已注册 [wandb]: origin 指本仓, setup 建 venv+装依赖。
`srv bootstrap wandb`(clone→setup→sync→start)或本机 `srv setup wandb && srv restart wandb`。
