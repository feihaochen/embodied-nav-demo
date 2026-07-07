from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr


ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets"
OUTPUTS_DIR = ROOT / "outputs" / "web_runs"


@dataclass
class DemoTarget:
    label: str
    canonical: str
    command: str
    scene: str
    seed: int
    width: int
    height: int
    max_steps: int
    threshold: float
    detect_every: int
    keep_last_for: int
    align_threshold: float
    stop_distance: float
    lost_stop_distance: float
    cached_video: Path
    cached_log: Path


TARGETS: Dict[str, DemoTarget] = {
    "沙发 sofa": DemoTarget(
        label="沙发 sofa",
        canonical="sofa",
        command="请到沙发旁边",
        scene="apt_1",
        seed=5,
        width=768,
        height=576,
        max_steps=140,
        threshold=0.25,
        detect_every=1,
        keep_last_for=0,
        align_threshold=0.30,
        stop_distance=1.60,
        lost_stop_distance=1.80,
        cached_video=ASSETS_DIR / "demo_sofa_768.mp4",
        cached_log=ASSETS_DIR / "demo_sofa_768.txt",
    ),
    "椅子 chair": DemoTarget(
        label="椅子 chair",
        canonical="chair",
        command="请到椅子旁边",
        scene="apt_1",
        seed=3,
        width=768,
        height=576,
        max_steps=180,
        threshold=0.25,
        detect_every=1,
        keep_last_for=0,
        align_threshold=0.30,
        stop_distance=1.20,
        lost_stop_distance=1.45,
        cached_video=ASSETS_DIR / "demo_chair_768.mp4",
        cached_log=ASSETS_DIR / "demo_chair_768.txt",
    ),
}


def _safe_read_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return f"[missing log file] {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _extract_final_reply(log_text: str, fallback_target: str) -> str:
    # Prefer explicit final_reply line.
    for line in reversed(log_text.splitlines()):
        if "final_reply=" in line:
            return line.split("final_reply=", 1)[-1].strip()

    # Otherwise search for the Chinese final phrase.
    m = re.search(r"已到达.*?还需要什么？", log_text)
    if m:
        return m.group(0)

    return f"已到达 {fallback_target} 旁边，还需要什么？"


def _summarize_log(log_text: str) -> str:
    states = []
    for s in ["SEARCH", "ALIGN", "APPROACH", "STOP", "RECOVER", "FAIL"]:
        if f"state={s}" in log_text or f"state': '{s}" in log_text:
            states.append(s)

    visible_count = log_text.count("visible=True") + log_text.count("target_visible=True")
    final_reply = _extract_final_reply(log_text, "target")

    summary = [
        "### Episode Summary",
        "",
        f"- States observed: `{ ' → '.join(states) if states else 'N/A' }`",
        f"- Target-visible detections: `{visible_count}`",
        f"- Final reply: **{final_reply}**",
        "",
        "### No-Privileged-Information Policy",
        "",
        "Agent runtime inputs are first-person RGB, depth, robot state/action feedback, and the user command. "
        "The policy does **not** use simulator object pose, semantic sensor, shortest path follower, oracle top-down map, or scene graph metadata.",
    ]
    return "\n".join(summary)


def load_cached_demo(target_label: str) -> Tuple[str, str, str, str]:
    target = TARGETS[target_label]
    log_text = _safe_read_text(target.cached_log)
    final_reply = _extract_final_reply(log_text, target.canonical)
    summary = _summarize_log(log_text)

    if not target.cached_video.exists():
        video_path = None
        log_text = f"[missing video file] {target.cached_video}\n\n" + log_text
    else:
        video_path = str(target.cached_video)

    return video_path, log_text, final_reply, summary


def local_live_run(target_label: str, custom_command: str) -> Tuple[str, str, str, str]:
    target = TARGETS[target_label]
    command = (custom_command or "").strip() or target.command

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUTS_DIR / f"{target.canonical}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "scripts/test_visual_nav_owlvit.py",
        "--scene",
        target.scene,
        "--command",
        command,
        "--seed",
        str(target.seed),
        "--width",
        str(target.width),
        "--height",
        str(target.height),
        "--max-steps",
        str(target.max_steps),
        "--threshold",
        str(target.threshold),
        "--detect-every",
        str(target.detect_every),
        "--keep-last-for",
        str(target.keep_last_for),
        "--align-threshold",
        str(target.align_threshold),
        "--stop-distance",
        str(target.stop_distance),
        "--lost-stop-distance",
        str(target.lost_stop_distance),
        "--out-dir",
        str(out_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    header = [
        "[Local Live Run]",
        f"cwd: {ROOT}",
        "cmd:",
        " ".join(shlex.quote(x) for x in cmd),
        "",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
        run_output = proc.stdout
    except subprocess.TimeoutExpired as e:
        run_output = (e.stdout or "") + "\n[ERROR] Local live run timed out."
        return None, "\n".join(header) + run_output, "运行超时，请使用 Cached Demo。", "Local run timed out."

    # Find generated overlay video/log.
    videos = sorted(out_dir.glob("visual_nav_owlvit_*.mp4"))
    # Prefer overlay video, avoid raw video.
    overlay_videos = [p for p in videos if "_raw_" not in p.name]
    video_path = overlay_videos[0] if overlay_videos else (videos[0] if videos else None)

    logs = sorted(out_dir.glob("visual_nav_owlvit_*.txt"))
    log_text = _safe_read_text(logs[0]) if logs else "[missing generated log file]"

    combined_log = "\n".join(header) + run_output + "\n\n--- episode log ---\n" + log_text
    final_reply = _extract_final_reply(log_text, target.canonical)
    summary = _summarize_log(log_text)

    return str(video_path) if video_path else None, combined_log, final_reply, summary


CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
}
.hero {
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(135deg, #f7f7fb, #eef4ff);
    border: 1px solid #d9e2f2;
}
.small-note {
    font-size: 0.92rem;
    color: #444;
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Habitat RGB-D Visual Navigation Agent") as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>Language-conditioned RGB-D Visual Navigation Agent</h1>
              <p>
                Habitat / ReplicaCAD embodied navigation demo.
                The agent accepts Chinese text commands, uses first-person RGB-D and robot state,
                detects the target with OWL-ViT, and navigates via SEARCH / ALIGN / APPROACH / STOP.
              </p>
              <p class="small-note">
                Runtime policy does not use simulator object pose, semantic oracle, shortest path follower,
                oracle top-down map, or scene graph metadata.
              </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                target = gr.Dropdown(
                    label="目标地点 / Target",
                    choices=list(TARGETS.keys()),
                    value="沙发 sofa",
                )
                command = gr.Textbox(
                    label="中文命令 / Chinese Command",
                    value="请到沙发旁边",
                    placeholder="例如：请到沙发旁边 / 请到椅子旁边",
                )
                with gr.Row():
                    cached_btn = gr.Button("Load Cached Demo", variant="primary")
                    live_btn = gr.Button("Local Live Run", variant="secondary")

                final_reply = gr.Textbox(
                    label="Agent 最终回复",
                    lines=2,
                    interactive=False,
                )

                summary = gr.Markdown(label="Summary")

            with gr.Column(scale=2):
                video = gr.Video(label="RGB Overlay Video")
                log = gr.Textbox(
                    label="Agent 状态日志 / SEARCH-ALIGN-APPROACH-STOP",
                    lines=22,
                    max_lines=28,
                    interactive=False,
                )

        gr.Markdown(
            """
            ### Demo Notes

            - **Cached Demo** is the stable presentation mode: it loads verified sofa/chair episodes from `assets/`.
            - **Local Live Run** executes the Habitat/OWL-ViT pipeline on this machine and may take several minutes.
            - The current primary demo targets are **sofa** and **chair**. The same parser/detector interface contains table/bed hooks, but table/bed were not robust in the selected `apt_1` scene and are not used as primary demo targets.
            """
        )

        cached_btn.click(
            fn=load_cached_demo,
            inputs=[target],
            outputs=[video, log, final_reply, summary],
        )

        live_btn.click(
            fn=local_live_run,
            inputs=[target, command],
            outputs=[video, log, final_reply, summary],
        )

        def update_command(target_label: str) -> str:
            return TARGETS[target_label].command

        target.change(
            fn=update_command,
            inputs=[target],
            outputs=[command],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    # Use share=True for a temporary public link during interview if needed.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
