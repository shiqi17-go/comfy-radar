#!/usr/bin/env python3
"""ComfyRadar 数据抓取脚本。

抓取来源：
  1. Civitai API      —— Checkpoint / LoRA 最新发布 + 热门
  2. ComfyUI Registry —— 插件（自定义节点）最新更新 + 安装量

输出：
  data/site.json                 —— 前端直接消费的聚合数据
  data/history/models-YYYY-MM-DD.json —— 每日快照（id -> 下载量），用于算增速

仅使用 Python 标准库；自动遵循 HTTPS_PROXY / HTTP_PROXY 环境变量。
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
SITE_JSON = DATA_DIR / "site.json"

UA = {"User-Agent": "ComfyRadar/0.1 (+https://github.com/)"}
TIMEOUT = 30


def http_get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------- Civitai ----------------

def _pick_gen_params(meta):
    """从 Civitai 示例图 meta 里提取生成参数。"""
    if not isinstance(meta, dict):
        return {}
    keys = ["prompt", "negativePrompt", "sampler", "steps", "cfgScale", "seed", "Size", "Model"]
    out = {}
    for k in keys:
        v = meta.get(k)
        if v is not None and v != "":
            out[k] = v
    # sampler 有时藏在其它字段
    if "sampler" not in out:
        for k in ("samplerName", "Sampler"):
            if meta.get(k):
                out["sampler"] = meta[k]
                break
    return out


def _strip_html(s):
    """去掉 Civitai 简介里的 HTML 标签，保留纯文本。"""
    if not s:
        return ""
    s = re.sub(r"<\s*br\s*/?\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&nbsp;?", " ", s)
    s = re.sub(r"&amp;", "&", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


# 按底座模型估算显存需求（经验值，供参考）
VRAM_TABLE = [
    (("wan", "video", "hunyuan", "ltx", "mochi", "cogvideo"), "约 12–24GB（视频生成模型，很吃显存）"),
    (("flux.1 d", "flux.1 dev", "flux dev"), "约 12–16GB（GGUF 量化版可降到 6–8GB）"),
    (("flux",), "约 12–16GB（GGUF 量化版可降到 6–8GB）"),
    (("sd3", "sd 3"), "约 8–12GB"),
    (("sdxl", "pony", "illustrious", "noobai"), "约 6–8GB"),
    (("sd 1", "sd1", "1.5"), "约 4GB，核显轻薄本也能跑"),
    (("qwen",), "约 12–16GB"),
]


def vram_hint(m):
    """给每个模型一句显存建议。"""
    if m.get("type") != "Checkpoint":
        return "随底座模型，LoRA 本身几乎不占额外显存"
    base = (m.get("baseModel") or "").lower()
    for keys, hint in VRAM_TABLE:
        if any(k in base for k in keys):
            return hint
    size = m.get("fileSizeMB") or 0
    if size > 8000:
        return "文件较大，建议 12GB 以上显存"
    return "约 6–8GB"


def usage_hint(m):
    """文件应该放进 ComfyUI 的哪个目录。"""
    t = m.get("type")
    return {
        "Checkpoint": "ComfyUI/models/checkpoints",
        "LORA": "ComfyUI/models/loras",
    }.get(t, "ComfyUI/models 对应子目录")


# LoRA 规则分类器：无需 AI，按名字+标签关键词自动归类
LORA_CAT_RULES = [
    ("工具增强", ["upscale", "detailer", "detail", "enhance", "fix", "hand fix",
                  "deblur", "sharpen", "quality", "hd", "anti-blur", "noise", "clarity"]),
    ("姿势动作", ["pose", "action", "gesture", "perspective", "angle", "sitting",
                  "standing", "lying", "fighting", "holding", "hug", "kiss"]),
    ("服装", ["dress", "outfit", "costume", "clothing", "uniform", "bikini", "kimono",
              "armor", "suit", "hat", "cosplay", "skirt", "hoodie", "jacket"]),
    ("角色", ["character", " oc", "original character", "girl", "boy", "waifu",
              "husbando", "celebrity", "vtuber", "persona", "my oc"]),
    ("身体五官", ["face", "eyes", "skin", "body", "breast", "abs", "muscle",
                  "hair", "smile", "expression", "makeup", "lips"]),
    ("场景背景", ["background", "scenery", "landscape", "city", "room", "indoor",
                  "outdoor", "nature", "lighting", "atmosphere", "night", "rain"]),
    ("风格", ["style", "painting", "watercolor", "ghibli", "pixel", "flat",
              "illustration", "sketch", "comic", "manga", "anime", "3d", "realistic",
              "cyberpunk", "retro", "vintage"]),
    ("概念特效", ["effect", "filter", "film", "photo", "bokeh", "vfx", "material",
                  "texture", "product", "object", "vehicle", "weapon", "food", "animal"]),
]


def classify_lora(m):
    """给 LoRA 自动分类，返回中文类名。"""
    if m.get("type") != "LORA":
        return ""
    text = " ".join([m.get("name") or "",
                     " ".join(m.get("tags") or []),
                     m.get("description") or ""]).lower()
    for cat, keywords in LORA_CAT_RULES:
        if any(k in text for k in keywords):
            return cat
    return "概念特效"


def _norm_model(item):
    ver = (item.get("modelVersions") or [{}])[0]
    images = ver.get("images") or []
    imgs = []
    params = {}
    for im in images[:4]:
        url = im.get("url")
        if url:
            imgs.append(url)
        if not params:
            params = _pick_gen_params(im.get("meta"))
    files = ver.get("files") or [{}]
    f0 = files[0] if files else {}
    stats = item.get("stats") or {}
    size_kb = f0.get("sizeKB")
    m = {
        "id": item.get("id"),
        "versionId": ver.get("id"),
        "name": item.get("name"),
        "type": item.get("type"),  # Checkpoint / LORA ...
        "creator": (item.get("creator") or {}).get("username", ""),
        "url": f"https://civitai.com/models/{item.get('id')}",
        "downloadUrl": ver.get("downloadUrl") or f"https://civitai.com/models/{item.get('id')}",
        "baseModel": ver.get("baseModel", ""),
        "versionName": ver.get("name", ""),
        "fileSizeMB": round(size_kb / 1024, 1) if size_kb else None,
        "image": imgs[0] if imgs else None,
        "images": imgs,
        "params": params,
        "description": _strip_html(item.get("description"))[:600],
        "trainedWords": (ver.get("trainedWords") or [])[:5],
        "downloads": stats.get("downloadCount", 0),
        "favorites": stats.get("favoriteCount", 0),
        "rating": stats.get("rating", 0),
        "publishedAt": ver.get("publishedAt") or item.get("lastVersionAt"),
        "tags": item.get("tags") or [],
        "growth": 0,  # 后面用快照回填
    }
    m["vram"] = vram_hint(m)
    m["usageDir"] = usage_hint(m)
    m["category"] = classify_lora(m)
    return m


def fetch_civitai():
    """抓最新 + 热门两批，合并去重。"""
    results = {}
    queries = [
        # (sort, period) 最新发布
        ("Newest", None),
        # 本周热门
        ("Highest Rated", "Week"),
        ("Most Downloaded", "Week"),
    ]
    for sort, period in queries:
        for mtype in ("Checkpoint", "LORA"):
            qs = urllib.parse.urlencode({
                "limit": 40, "types": mtype, "sort": sort,
                "nsfw": "false", **({"period": period} if period else {}),
            })
            url = f"https://civitai.com/api/v1/models?{qs}"
            try:
                data = http_get_json(url)
            except Exception as e:
                print(f"[civitai] {sort}/{mtype} 抓取失败: {e}", file=sys.stderr)
                continue
            for item in data.get("items", []):
                m = _norm_model(item)
                if m["id"] is not None:
                    results[m["id"]] = m
            time.sleep(1)  # 礼貌限速
    models = list(results.values())
    enrich_gen_params(models)
    return models


def enrich_gen_params(models, max_calls=60):
    """列表接口不返回生成参数，需用 images 接口 + withMeta=true 单独补。

    优先补下载量高的模型（热门模型的参数更有参考价值）。
    """
    todo = sorted(models, key=lambda m: -(m["downloads"] or 0))[:max_calls]
    ok = 0
    for m in todo:
        if not m.get("versionId"):
            continue
        qs = urllib.parse.urlencode({
            "modelVersionId": m["versionId"], "limit": 4,
            "nsfw": "false", "withMeta": "true",
        })
        try:
            data = http_get_json(f"https://civitai.com/api/v1/images?{qs}")
        except Exception as e:
            print(f"[civitai] 参数补抓失败 model={m['id']}: {e}", file=sys.stderr)
            continue
        imgs, params = [], {}
        for im in data.get("items", []):
            if im.get("url"):
                imgs.append(im["url"])
            if not params:
                params = _pick_gen_params(im.get("meta"))
        if imgs:
            m["images"] = imgs
            m["image"] = imgs[0]
        if params:
            m["params"] = params
            ok += 1
        time.sleep(0.5)
    print(f"    生成参数补抓完成：{ok}/{len(todo)} 个模型有参数")


COVERS_DIR = DATA_DIR / "covers"


def download_covers(models, max_calls=120):
    """把封面图下载到本地 data/covers/，国内直连 Civitai 图床失败时自动熔断。

    部署在 GitHub Actions 上跑时国外网络可正常下载；
    国内本地运行时熔断后保留远程 URL，页面自动隐藏加载失败的图。
    """
    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    ok, fails = 0, 0
    for m in models[:max_calls]:
        url = m.get("image")
        if not url:
            continue
        dst = COVERS_DIR / f"{m['id']}.jpg"
        if dst.exists() and dst.stat().st_size > 1000:
            m["imageLocal"] = f"data/covers/{m['id']}.jpg"
            ok += 1
            continue
        thumb = url.replace("/original=true/", "/width=450/")
        try:
            req = urllib.request.Request(thumb, headers=UA)
            data = urllib.request.urlopen(req, timeout=10).read()
            if len(data) > 1000:
                dst.write_bytes(data)
                m["imageLocal"] = f"data/covers/{m['id']}.jpg"
                ok += 1
                fails = 0
            else:
                fails += 1
        except Exception:
            fails += 1
        if fails >= 5:
            print("    图床不可达（国内网络常见），跳过封面本地化，部署后在 Actions 里会自动补上",
                  file=sys.stderr)
            break
        time.sleep(0.3)
    print(f"    封面本地化：{ok} 张")


# ---------------- ComfyUI Registry ----------------

def fetch_registry():
    """ComfyUI 官方插件注册中心。失败时退回 ComfyUI-Manager 的列表。"""
    nodes = {}
    try:
        page = 1
        while page <= 5:  # 最多抓 500 个
            data = http_get_json(
                f"https://api.comfy.org/nodes?page={page}&limit=100")
            items = data.get("nodes") or data.get("items") or []
            if not items:
                break
            for n in items:
                latest = n.get("latest_version") or {}
                nodes[n.get("id")] = {
                    "id": n.get("id"),
                    "name": n.get("name") or n.get("id"),
                    "author": n.get("author") or "",
                    "description": (n.get("description") or "")[:200],
                    "installs": n.get("downloads", 0),
                    "stars": n.get("github_stars", 0) or 0,
                    "version": latest.get("version") or "",
                    "url": f"https://registry.comfy.org/nodes/{n.get('id')}",
                    "repo": n.get("repository") or "",
                    "updatedAt": latest.get("createdAt") or n.get("createdAt"),
                }
            page += 1
            time.sleep(1)
        if nodes:
            return list(nodes.values())
    except Exception as e:
        print(f"[registry] 官方 API 抓取失败: {e}，尝试 Manager 列表", file=sys.stderr)

    # 兜底：ComfyUI-Manager 的扩展列表（无安装量数据）
    try:
        data = http_get_json(
            "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json")
        for repo_url, info in list(data.items())[:500]:
            key = repo_url.rstrip("/").split("/")[-1]
            nodes[key] = {
                "id": key,
                "name": key,
                "author": repo_url.rstrip("/").split("/")[-2],
                "description": "",
                "installs": 0,
                "stars": 0,
                "version": "",
                "url": repo_url,
                "repo": repo_url,
                "updatedAt": None,
            }
    except Exception as e:
        print(f"[registry] Manager 列表也失败: {e}", file=sys.stderr)
    return list(nodes.values())


# ---------------- 增速计算 ----------------

def today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_latest_snapshot():
    if not HISTORY_DIR.exists():
        return None, None
    files = sorted(HISTORY_DIR.glob("models-*.json"))
    if not files:
        return None, None
    latest = files[-1]
    try:
        return latest.stem.replace("models-", ""), json.loads(
            latest.read_text(encoding="utf-8"))
    except Exception:
        return None, None


def apply_growth(models):
    """用最近一次快照计算下载量增量。同一天重复抓取不算增速。"""
    snap_date, snapshot = load_latest_snapshot()
    today = today_str()
    if snapshot and snap_date != today:
        for m in models:
            old = snapshot.get(str(m["id"]))
            if old is not None and m["downloads"] is not None:
                m["growth"] = max(0, m["downloads"] - old)
    # 保存今日快照
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    snap = {str(m["id"]): m["downloads"] for m in models if m["downloads"] is not None}
    (HISTORY_DIR / f"models-{today}.json").write_text(
        json.dumps(snap, ensure_ascii=False), encoding="utf-8")


# ---------------- 演示数据 ----------------

def demo_data():
    return {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "demo": True,
        "models": [
            {
                "id": 1001, "name": "PixelWave XL", "type": "Checkpoint",
                "creator": "demo_author", "url": "https://civitai.com/models/1001",
                "downloadUrl": "https://civitai.com/api/download/models/1001",
                "baseModel": "SDXL 1.0", "versionName": "v2.0", "fileSizeMB": 6500.0,
                "image": None, "images": [],
                "params": {"prompt": "masterpiece, best quality, 1girl, city night",
                           "negativePrompt": "lowres, bad anatomy",
                           "sampler": "DPM++ 2M Karras", "steps": 28, "cfgScale": 7,
                           "seed": 123456789},
                "description": "一个赛博朋克风格的 SDXL 大模型，擅长夜景和霓虹光效。",
                "trainedWords": ["pixelwave", "neon city"],
                "vram": "约 6–8GB", "usageDir": "ComfyUI/models/checkpoints",
                "downloads": 15230, "favorites": 890, "rating": 4.8,
                "publishedAt": datetime.now(timezone.utc).isoformat(),
                "tags": ["anime", "style"], "growth": 3200,
            },
            {
                "id": 1002, "name": "Product Photo LoRA", "type": "LORA",
                "creator": "studio_ai", "url": "https://civitai.com/models/1002",
                "downloadUrl": "https://civitai.com/api/download/models/1002",
                "baseModel": "Flux.1 D", "versionName": "v1.1", "fileSizeMB": 228.0,
                "image": None, "images": [],
                "params": {"prompt": "product photography, studio lighting, <lora:product:0.8>",
                           "sampler": "Euler", "steps": 20, "cfgScale": 3.5},
                "description": "电商产品图专用 LoRA，一键生成影棚级打光效果。",
                "trainedWords": ["product photo"],
                "vram": "随底座模型，LoRA 本身几乎不占额外显存",
                "usageDir": "ComfyUI/models/loras",
                "downloads": 8200, "favorites": 410, "rating": 4.6,
                "publishedAt": (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
                "tags": ["product", "photography"], "growth": 5100,
            },
        ],
        "plugins": [
            {
                "id": "comfyui-manager", "name": "ComfyUI-Manager",
                "author": "ltdrdata",
                "description": "安装、更新、管理自定义节点与模型",
                "installs": 5200000, "stars": 9800, "version": "v3.31",
                "url": "https://registry.comfy.org/nodes/comfyui-manager",
                "repo": "https://github.com/ltdrdata/ComfyUI-Manager",
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": "comfyui-videohelpersuite", "name": "ComfyUI-VideoHelperSuite",
                "author": "Kosinkadink",
                "description": "Nodes related to video workflows",
                "installs": 3400000, "stars": 1800, "version": "v1.7.9",
                "url": "https://registry.comfy.org/nodes/comfyui-videohelpersuite",
                "repo": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
                "updatedAt": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            },
        ],
    }


# ---------------- 主流程 ----------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    demo = "--demo" in sys.argv
    if demo:
        data = demo_data()
    else:
        print("[*] 抓取 Civitai ...")
        models = fetch_civitai()
        print(f"    模型/LoRA 共 {len(models)} 条")
        print("[*] 抓取 ComfyUI Registry ...")
        plugins = fetch_registry()
        print(f"    插件共 {len(plugins)} 条")
        apply_growth(models)
        data = {
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "demo": False,
            "models": models,
            "plugins": plugins,
        }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 保护：两个源都失败时保留旧数据，不要用空数据覆盖
    if not demo and not data["models"] and not data["plugins"] and SITE_JSON.exists():
        print("[!] 所有数据源都失败，保留现有 site.json 不覆盖", file=sys.stderr)
        return
    SITE_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"[OK] 写入 {SITE_JSON}（模型 {len(data['models'])} / 插件 {len(data['plugins'])}）")
    # 数据落盘后再下载封面，即使中断也不丢数据
    if not demo:
        print("[*] 下载封面图 ...")
        try:
            download_covers(data["models"])
            SITE_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            print("[OK] 封面字段已写回 site.json")
        except Exception as e:
            print(f"[!] 封面下载中断（数据已保存，不影响网站）: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
