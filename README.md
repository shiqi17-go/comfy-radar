# ComfyRadar · ComfyUI 情报站

每天自动聚合 ComfyUI 生态的最新动态：模型 / LoRA / 插件 / 工作流 / 论文，中英混排，每条带下载地址和生成参数，附增速榜，帮你发现刚爆火的新东西。

## 数据来源（全部免费）

| 内容 | 来源 |
|---|---|
| Checkpoint / LoRA | [Civitai REST API](https://developer.civitai.com/site/) |
| 插件（自定义节点） | [ComfyUI Registry](https://registry.comfy.org/) / ComfyUI-Manager |
| 工作流（二期） | comfy.org workflows |
| 论文（二期） | HuggingFace Papers / arXiv |

## 目录结构

```
comfy-radar/
├── index.html          # 前端（纯静态，零构建）
├── data/
│   ├── site.json       # 网站展示用的聚合数据（脚本生成）
│   └── history/        # 每日快照，用于计算增速
├── scripts/
│   └── fetch_all.py    # 抓取脚本（Python 3.8+，零依赖，只用标准库）
└── .github/workflows/
    └── daily.yml       # GitHub Actions 每日定时抓取
```

## 本地运行

```bash
# 1. 抓取数据（国内机器访问 Civitai 需要代理，脚本自动读取 HTTPS_PROXY 环境变量）
python scripts/fetch_all.py

# 网络不通时可以用演示数据先看效果
python scripts/fetch_all.py --demo

# 2. 起本地服务看网站
python -m http.server 8080
# 打开 http://127.0.0.1:8080
```

## 部署（零成本）

1. 把本目录推到 GitHub 仓库
2. Settings → Pages → 选 `main` 分支根目录
3. Actions 里的 `daily.yml` 会每天自动抓数据并提交，网站自动更新

## 路线图

- [x] 一期：模型/LoRA + 插件 + 下载地址 + 生成参数 + 增速榜
- [ ] 二期：工作流、论文、GitHub 增速
- [ ] 三期：RSS/邮件订阅、本地 ComfyUI 联动（检测已装模型/插件的更新）
