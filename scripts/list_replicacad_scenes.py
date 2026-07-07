from pathlib import Path
import os

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat_sim


def main():
    cfg_path = Path("data/replica_cad/replicaCAD.scene_dataset_config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing {cfg_path}. Did you download replica_cad_dataset?"
        )

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(cfg_path)
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    try:
        handles = list(sim.metadata_mediator.get_scene_handles())
        handles = [h for h in handles if h and h != "NONE"]

        print("[OK] ReplicaCAD scene handles:")
        for h in handles:
            print("  ", h)

        print(f"\n[OK] Total scenes: {len(handles)}")
        if handles:
            print("\nExample:")
            print(f"  python scripts/smoke_replicacad_nav.py --scene \"{handles[0]}\"")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
