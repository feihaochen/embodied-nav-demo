import argparse
from pathlib import Path

import imageio.v2 as imageio

from src.agent.owlvit_detector import OwlVitDetector, OwlVitDetectorConfig
from src.sim.habitat_backend import HabitatBackend, HabitatBackendConfig
from src.utils.vis import draw_debug_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="apt_0")
    parser.add_argument("--target", type=str, default="sofa")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            detect_every=1,
            keep_last_for=0,
        )
    )

    best = None
    all_overlay_frames = []

    try:
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            print(f"\n[SCAN] scene={args.scene}, seed={seed}, target={args.target}")
            detector.reset()
            obs = backend.reset(scene=args.scene, seed=seed)

            # One 360-degree scan. If your turn angle is 15 degrees, 24 turns ≈ 360 degrees.
            for step in range(24):
                det = detector.detect(obs["rgb"], args.target)

                debug = {
                    "state": "SCAN",
                    "action": "turn_left",
                    "target": args.target,
                    "target_visible": det is not None,
                }

                if det is not None:
                    debug.update(
                        {
                            "bbox": det.bbox_xyxy,
                            "score": det.score,
                            "label": det.label,
                        }
                    )
                    print(
                        f"[FOUND] seed={seed} step={step} "
                        f"score={det.score:.3f} bbox={det.bbox_xyxy} label={det.label}"
                    )

                    if best is None or det.score > best["score"]:
                        best = {
                            "seed": seed,
                            "step": step,
                            "score": det.score,
                            "bbox": det.bbox_xyxy,
                            "rgb": obs["rgb"].copy(),
                            "debug": debug.copy(),
                        }

                overlay = draw_debug_overlay(
                    obs["rgb"],
                    debug=debug,
                    reply=f"scanning for {args.target}",
                )
                all_overlay_frames.append(overlay)

                obs = backend.step("turn_left")

        scan_video = out_dir / f"scan_{args.scene}_{args.target}.mp4"
        imageio.mimsave(scan_video, all_overlay_frames, fps=8)
        print("[OK] saved scan video:", scan_video)

        if best is not None:
            best_img = draw_debug_overlay(
                best["rgb"],
                debug=best["debug"],
                reply=f"best detection seed={best['seed']} step={best['step']}",
            )
            best_path = out_dir / f"best_{args.scene}_{args.target}.png"
            imageio.imwrite(best_path, best_img)
            print("[OK] best detection:")
            print("  seed:", best["seed"])
            print("  step:", best["step"])
            print("  score:", best["score"])
            print("  bbox:", best["bbox"])
            print("  image:", best_path)
        else:
            print("[WARN] no detection found. Try another scene, more seeds, or lower threshold.")

    finally:
        backend.close()


if __name__ == "__main__":
    main()
