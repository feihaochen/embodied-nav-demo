import argparse
from pathlib import Path

import imageio.v2 as imageio

from src.agent.command_parser import parse_command
from src.agent.depth_safe_search_agent import DepthSafeSearchAgent
from src.agent.owlvit_detector import OwlVitDetector, OwlVitDetectorConfig
from src.sim.habitat_backend import HabitatBackend, HabitatBackendConfig
from src.utils.vis import draw_debug_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="apt_0")
    parser.add_argument("--command", type=str, default="请到沙发旁边")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--keep-last-for", type=int, default=0)
    parser.add_argument("--align-threshold", type=float, default=0.30)
    parser.add_argument("--stop-distance", type=float, default=1.60)
    parser.add_argument("--lost-stop-distance", type=float, default=1.80)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_command(args.command)
    print("[INFO] command:", args.command)
    print("[INFO] parsed:", parsed)

    if not parsed.ok:
        raise ValueError(f"Command parse failed: {parsed.message}")

    backend = HabitatBackend(
        HabitatBackendConfig(
            scene=args.scene,
            width=args.width,
            height=args.height,
            gpu_device_id=-1,
        )
    )

    detector = OwlVitDetector(
        OwlVitDetectorConfig(
            threshold=args.threshold,
            detect_every=args.detect_every,
            keep_last_for=args.keep_last_for,
        )
    )

    agent = DepthSafeSearchAgent(
        detector=detector,
        max_steps=args.max_steps,
        front_obstacle_threshold_m=0.55,
        stop_distance_m=args.stop_distance,
        lost_target_stop_distance_m=args.lost_stop_distance,
        align_threshold=args.align_threshold,
    )

    overlay_frames = []
    logs = []

    try:
        obs = backend.reset(scene=args.scene, seed=args.seed)
        detector.reset()
        agent.reset()

        logs.append(f"scene={args.scene}")
        logs.append(f"seed={args.seed}")
        logs.append(f"command={args.command}")
        logs.append(f"target={parsed.target}")
        logs.append("")

        final_reply = ""

        for step in range(args.max_steps):
            output = agent.act(obs, parsed)
            debug = output.debug
            final_reply = output.reply

            line = (
                f"step={step:03d} "
                f"state={debug.get('state')} "
                f"action={output.action} "
                f"target={debug.get('target')} "
                f"visible={debug.get('target_visible')} "
                f"front_depth={debug.get('front_depth')} "
                f"target_distance={debug.get('target_distance')} "
                f"bbox={debug.get('bbox')} "
                f"score={debug.get('score')} "
                f"reply={output.reply}"
            )
            print(line)
            logs.append(line)

            overlay = draw_debug_overlay(
                obs["rgb"],
                debug=debug,
                reply=output.reply,
            )
            overlay_frames.append(overlay)

            if output.done or output.action == "stop":
                logs.append("")
                logs.append(f"final_reply={output.reply}")
                print("[DONE]", output.reply)
                break

            obs = backend.step(output.action)

        suffix = f"{args.scene}_{parsed.target}_seed{args.seed}"
        overlay_path = out_dir / f"visual_nav_owlvit_{suffix}.mp4"
        raw_path = out_dir / f"visual_nav_owlvit_raw_{suffix}.mp4"
        log_path = out_dir / f"visual_nav_owlvit_{suffix}.txt"

        imageio.mimsave(overlay_path, overlay_frames, fps=8)
        backend.save_rgb_video(str(raw_path), fps=8)
        log_path.write_text("\n".join(logs), encoding="utf-8")

        print("[OK] saved overlay video:", overlay_path)
        print("[OK] saved raw video:", raw_path)
        print("[OK] saved log:", log_path)
        print("[FINAL_REPLY]", final_reply)

    finally:
        backend.close()


if __name__ == "__main__":
    main()
