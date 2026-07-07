import argparse
from pathlib import Path

from src.sim.habitat_backend import HabitatBackend, HabitatBackendConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="apt_0")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    cfg = HabitatBackendConfig(
        scene=args.scene,
        width=args.width,
        height=args.height,
        gpu_device_id=-1,  # Do not change on your current WSL2 setup.
    )
    backend = HabitatBackend(cfg)

    try:
        print("[INFO] Available scenes, first 10:")
        scenes = backend.list_scenes()
        for s in scenes[:10]:
            print("  ", s)

        print(f"[INFO] Reset scene={args.scene}, seed={args.seed}")
        obs = backend.reset(scene=args.scene, seed=args.seed)
        print("[INFO] First observation:")
        print("  scene:", obs["scene"])
        print("  rgb shape:", obs["rgb"].shape)
        print("  depth shape:", obs["depth"].shape)
        print("  agent_position:", obs["agent_position"])

        actions = []
        actions += ["turn_left"] * 24
        actions += ["move_forward"] * 12
        actions += ["turn_right"] * 12
        actions += ["move_forward"] * 12

        for i, action in enumerate(actions):
            obs = backend.step(action)
            if i % 10 == 0:
                print(
                    f"[STEP {i:03d}] action={action}, "
                    f"collided={obs['collided']}, "
                    f"pos={obs['agent_position']}"
                )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = out_dir / "m1_backend_random_walk_rgb.mp4"
        depth_path = out_dir / "m1_backend_random_walk_depth.mp4"

        backend.save_rgb_video(str(rgb_path), fps=10)
        backend.save_depth_video(str(depth_path), fps=10)

        print("[OK] Saved RGB video:", rgb_path)
        print("[OK] Saved Depth video:", depth_path)

    finally:
        backend.close()


if __name__ == "__main__":
    main()
