import argparse
from pathlib import Path

from src.agent.command_parser import parse_command
from src.agent.depth_safe_search_agent import DepthSafeSearchAgent
from src.sim.habitat_backend import HabitatBackend, HabitatBackendConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="apt_0")
    parser.add_argument("--command", type=str, default="请到沙发旁边")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_command(args.command)
    print("[INFO] command:", args.command)
    print("[INFO] parsed:", parsed)

    cfg = HabitatBackendConfig(
        scene=args.scene,
        width=args.width,
        height=args.height,
        gpu_device_id=-1,  # Critical for your current setup.
    )
    backend = HabitatBackend(cfg)
    agent = DepthSafeSearchAgent(max_steps=args.max_steps)

    logs = []

    try:
        obs = backend.reset(scene=args.scene, seed=args.seed)
        agent.reset()

        logs.append(f"command={args.command}")
        logs.append(f"parsed_target={parsed.target}")
        logs.append(f"scene={obs['scene']}")
        logs.append("")

        for step in range(args.max_steps):
            output = agent.act(obs, parsed)

            debug = output.debug
            line = (
                f"step={step:03d} "
                f"state={debug.get('state')} "
                f"action={output.action} "
                f"target={debug.get('target')} "
                f"visible={debug.get('target_visible')} "
                f"front_depth={debug.get('front_depth')} "
                f"reply={output.reply}"
            )
            print(line)
            logs.append(line)

            if output.done or output.action == "stop":
                logs.append("")
                logs.append(f"final_reply={output.reply}")
                break

            obs = backend.step(output.action)

        rgb_path = out_dir / "m1_agent_loop_rgb.mp4"
        depth_path = out_dir / "m1_agent_loop_depth.mp4"
        log_path = out_dir / "m1_agent_loop_log.txt"

        backend.save_rgb_video(str(rgb_path), fps=10)
        backend.save_depth_video(str(depth_path), fps=10)
        log_path.write_text("\n".join(logs), encoding="utf-8")

        print("[OK] Saved RGB video:", rgb_path)
        print("[OK] Saved Depth video:", depth_path)
        print("[OK] Saved log:", log_path)

    finally:
        backend.close()


if __name__ == "__main__":
    main()
