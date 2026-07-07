from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.agent.command_parser import parse_command


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
RUNS_DIR = PROJECT_ROOT / "outputs" / "web_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Habitat RGB-D Visual Navigation Agent")

app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


DEMOS = {
    "sofa": {
        "label": "沙发 / sofa",
        "command": "请到沙发旁边",
        "scene": "apt_1",
        "seed": 5,
        "width": 768,
        "height": 576,
        "max_steps": 140,
        "threshold": 0.25,
        "detect_every": 1,
        "keep_last_for": 0,
        "align_threshold": 0.30,
        "stop_distance": 1.60,
        "lost_stop_distance": 1.80,
        "cached_video": "demo_sofa_768.mp4",
        "cached_log": "demo_sofa_768.txt",
        "reply": "已到达 sofa 旁边，还需要什么？",
    },
    "chair": {
        "label": "椅子 / chair",
        "command": "请到椅子旁边",
        "scene": "apt_1",
        "seed": 3,
        "width": 768,
        "height": 576,
        "max_steps": 180,
        "threshold": 0.25,
        "detect_every": 1,
        "keep_last_for": 0,
        "align_threshold": 0.30,
        "stop_distance": 1.20,
        "lost_stop_distance": 1.45,
        "cached_video": "demo_chair_768.mp4",
        "cached_log": "demo_chair_768.txt",
        "reply": "已到达 chair 旁边，还需要什么？",
    },
}


def read_text(path: Path, max_chars: int = 16000) -> str:
    if not path.exists():
        return f"[missing file] {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def extract_final_reply(log_text: str, fallback: str) -> str:
    for line in reversed(log_text.splitlines()):
        if "final_reply=" in line:
            return line.split("final_reply=", 1)[1].strip()
    return fallback


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Habitat RGB-D Visual Navigation Agent</title>
  <style>
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans SC", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
    }
    header {
      padding: 28px 24px 18px;
      text-align: center;
      background: linear-gradient(135deg, #172554, #0f172a);
      border-bottom: 1px solid #334155;
    }
    h1 { margin: 0 0 10px; font-size: 30px; }
    p { line-height: 1.65; color: #cbd5e1; }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 20px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.25fr 0.75fr;
      gap: 18px;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: #111827;
      border: 1px solid #334155;
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 18px 50px rgba(0,0,0,0.25);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
      align-items: center;
    }
    select, input {
      background: #020617;
      color: #e5e7eb;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 15px;
    }
    input { min-width: 260px; flex: 1; }
    button {
      background: #2563eb;
      color: white;
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 15px;
      cursor: pointer;
      font-weight: 700;
    }
    button.secondary { background: #0f766e; }
    video {
      width: 100%;
      background: black;
      border-radius: 14px;
      border: 1px solid #334155;
    }
    pre {
      background: #020617;
      color: #d1d5db;
      border: 1px solid #334155;
      border-radius: 14px;
      padding: 14px;
      max-height: 520px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.45;
    }
    .reply {
      margin: 12px 0;
      padding: 12px 14px;
      border-radius: 12px;
      background: #052e2b;
      border: 1px solid #14b8a6;
      color: #99f6e4;
      font-weight: 800;
    }
    .badge {
      display: inline-block;
      padding: 5px 9px;
      margin: 3px;
      border: 1px solid #334155;
      border-radius: 999px;
      background: #020617;
      color: #cbd5e1;
      font-size: 13px;
    }
    code {
      background: #020617;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 2px 6px;
    }
    .warn { color: #fbbf24; }
  </style>
</head>
<body>
<header>
  <h1>Habitat / ReplicaCAD 语言条件 RGB-D 视觉导航 Agent</h1>
  <p>
    用户输入“请到沙发旁边”或“请到椅子旁边”，Agent 基于第一视角 RGB-D、
    机器人本体状态和动作反馈执行 SEARCH / ALIGN / APPROACH / STOP 闭环导航。
  </p>
</header>

<main>
  <section class="grid">
    <div class="card">
      <div class="controls">
        <select id="target" onchange="onTargetChange()">
          <option value="sofa">沙发 / sofa</option>
          <option value="chair">椅子 / chair</option>
        </select>
        <input id="command" value="请到沙发旁边"/>
        <button onclick="loadCached()">Load Cached Demo</button>
        <button class="secondary" onclick="runLive()">Local Live Run</button>
      </div>

      <video id="video" controls playsinline src="/assets/demo_sofa_768.mp4"></video>
      <div id="reply" class="reply">已到达 sofa 旁边，还需要什么？</div>
      <p id="status">Cached Demo: sofa</p>
    </div>

    <div class="card">
      <h2>Agent 输入约束</h2>
      <p>运行时 Agent 只使用：</p>
      <span class="badge">RGB</span>
      <span class="badge">Depth</span>
      <span class="badge">Robot state</span>
      <span class="badge">Action feedback</span>
      <span class="badge">Text command</span>

      <p>Agent 不使用：</p>
      <span class="badge warn">object pose</span>
      <span class="badge warn">semantic oracle</span>
      <span class="badge warn">shortest path</span>
      <span class="badge warn">top-down oracle map</span>

      <h2>状态机</h2>
      <p><code>SEARCH → ALIGN → APPROACH → STOP</code></p>
    </div>
  </section>

  <section class="card" style="margin-top:18px;">
    <h2>状态日志</h2>
    <pre id="log">Loading...</pre>
  </section>
</main>

<script>
const DEMOS = {
  sofa: {
    command: "请到沙发旁边",
    video: "/assets/demo_sofa_768.mp4",
    log: "/assets/demo_sofa_768.txt",
    reply: "已到达 sofa 旁边，还需要什么？"
  },
  chair: {
    command: "请到椅子旁边",
    video: "/assets/demo_chair_768.mp4",
    log: "/assets/demo_chair_768.txt",
    reply: "已到达 chair 旁边，还需要什么？"
  }
};

function onTargetChange() {
  const key = document.getElementById("target").value;
  document.getElementById("command").value = DEMOS[key].command;
}

async function loadCached() {
  const key = document.getElementById("target").value;
  const d = DEMOS[key];

  document.getElementById("video").src = d.video;
  document.getElementById("video").load();
  document.getElementById("reply").innerText = d.reply;
  document.getElementById("status").innerText = "Cached Demo: " + key;

  try {
    const resp = await fetch(d.log);
    document.getElementById("log").innerText = await resp.text();
  } catch (e) {
    document.getElementById("log").innerText = "Failed to load cached log: " + e;
  }
}

async function runLive() {
  const key = document.getElementById("target").value;
  const command = document.getElementById("command").value;

  document.getElementById("status").innerText = "Running live episode... This may take several minutes.";
  document.getElementById("log").innerText = "Running Habitat + OWL-ViT local live episode...";

  try {
    const resp = await fetch("/api/live", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        target: key,
        command: command,
        resolution: "768x576"
      })
    });

    const data = await resp.json();

    if (!data.ok) {
      document.getElementById("status").innerText = "Live run failed.";
      document.getElementById("reply").innerText = data.reply || "运行失败";
      document.getElementById("log").innerText = data.log || JSON.stringify(data, null, 2);
      return;
    }

    document.getElementById("video").src = data.video_url;
    document.getElementById("video").load();
    document.getElementById("reply").innerText = data.reply;
    document.getElementById("status").innerText = data.status;
    document.getElementById("log").innerText = data.log;
  } catch (e) {
    document.getElementById("status").innerText = "Live run request failed.";
    document.getElementById("log").innerText = String(e);
  }
}

loadCached();
</script>
</body>
</html>
        """
    )


@app.get("/api/cached/{target}")
def api_cached(target: str):
    if target not in DEMOS:
        return JSONResponse({"ok": False, "error": f"Unknown target: {target}"}, status_code=400)

    d = DEMOS[target]
    log_path = ASSETS_DIR / d["cached_log"]
    log_text = read_text(log_path)
    reply = extract_final_reply(log_text, d["reply"])

    return {
        "ok": True,
        "target": target,
        "video_url": f"/assets/{d['cached_video']}",
        "log": log_text,
        "reply": reply,
    }


@app.post("/api/live")
async def api_live(request: Request):
    payload = await request.json()
    target = str(payload.get("target", "sofa"))
    command = str(payload.get("command", "")).strip()
    resolution = str(payload.get("resolution", "320x240"))

    if target not in DEMOS:
        return JSONResponse({"ok": False, "error": f"Unknown target: {target}"}, status_code=400)

    demo = DEMOS[target]
    if not command:
        command = demo["command"]

    parsed = parse_command(command)
    if not parsed.ok or parsed.target != target:
        return {
            "ok": False,
            "reply": "命令解析失败或与目标不一致",
            "log": f"command={command}\\nparsed={parsed}\\nexpected_target={target}",
        }

    if resolution == "768x576":
        width, height = 768, 576
    else:
        width, height = 320, 240

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / f"{target}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    script = PROJECT_ROOT / "scripts" / "test_visual_nav_owlvit.py"

    cmd = [
        sys.executable,
        str(script),
        "--scene", demo["scene"],
        "--command", command,
        "--seed", str(demo["seed"]),
        "--width", str(width),
        "--height", str(height),
        "--max-steps", str(demo["max_steps"]),
        "--threshold", str(demo["threshold"]),
        "--detect-every", str(demo["detect_every"]),
        "--keep-last-for", str(demo["keep_last_for"]),
        "--align-threshold", str(demo["align_threshold"]),
        "--stop-distance", str(demo["stop_distance"]),
        "--lost-stop-distance", str(demo["lost_stop_distance"]),
        "--out-dir", str(out_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1200,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "reply": "运行超时，请使用 Cached Demo",
            "log": (exc.stdout or "") + "\\n[ERROR] timeout",
        }

    expected_video = out_dir / f"visual_nav_owlvit_{demo['scene']}_{target}_seed{demo['seed']}.mp4"
    expected_log = out_dir / f"visual_nav_owlvit_{demo['scene']}_{target}_seed{demo['seed']}.txt"

    log_text = read_text(expected_log) if expected_log.exists() else ""
    full_log = (
        "===== subprocess output =====\\n"
        + proc.stdout[-12000:]
        + "\\n\\n===== episode log =====\\n"
        + log_text
    )

    if proc.returncode != 0:
        return {
            "ok": False,
            "reply": "Local Live Run 失败",
            "log": full_log,
        }

    if not expected_video.exists():
        return {
            "ok": False,
            "reply": "运行完成但未找到输出视频",
            "log": full_log + f"\\nMissing video: {expected_video}",
        }

    reply = extract_final_reply(log_text, demo["reply"])
    video_url = f"/runs/{out_dir.name}/{expected_video.name}"

    return {
        "ok": True,
        "target": target,
        "video_url": video_url,
        "reply": reply,
        "log": full_log,
        "status": f"Local Live Run finished: {target}, {width}x{height}",
    }
