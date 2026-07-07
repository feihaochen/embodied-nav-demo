import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat_sim
import magnum as mn
from habitat_sim.utils import common as utils


def make_camera(uuid, sensor_type, width, height):
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = uuid
    sensor.sensor_type = sensor_type
    sensor.resolution = [height, width]
    sensor.position = [0.0, 1.25, 0.0]
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    return sensor


def get_scene_handles():
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
        return handles
    finally:
        sim.close()


def resolve_scene_handle(requested):
    handles = get_scene_handles()

    if not handles:
        raise RuntimeError("No ReplicaCAD scene handles found.")

    if requested in handles:
        return requested

    # 允许用户只写 apt_0 / apt_1；如果 handle 是完整路径，则模糊匹配
    for h in handles:
        if requested and requested in h:
            print(f"[INFO] Matched requested scene '{requested}' to handle:")
            print("       ", h)
            return h

    print(f"[WARN] Requested scene '{requested}' not found.")
    print("[WARN] Available scenes:")
    for h in handles[:20]:
        print("  ", h)

    print("[WARN] Using first available scene:")
    print("  ", handles[0])
    return handles[0]


def make_sim(scene_handle, width=320, height=240):
    dataset_cfg = Path("data/replica_cad/replicaCAD.scene_dataset_config.json")
    if not dataset_cfg.exists():
        raise FileNotFoundError(
            "Missing data/replica_cad/replicaCAD.scene_dataset_config.json"
        )

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(dataset_cfg)
    sim_cfg.scene_id = scene_handle

    # 关键：Win10 WSL2 + source no-CUDA build 下，不要强绑 CUDA device 0
    sim_cfg.gpu_device_id = -1

    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = True

    rgb_sensor = make_camera(
        uuid="rgb",
        sensor_type=habitat_sim.SensorType.COLOR,
        width=width,
        height=height,
    )
    depth_sensor = make_camera(
        uuid="depth",
        sensor_type=habitat_sim.SensorType.DEPTH,
        width=width,
        height=height,
    )

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.20),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def to_rgb(obs):
    rgb = obs["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb.astype(np.uint8)


def depth_vis(obs, max_depth=10.0):
    depth = obs["depth"]
    depth = np.clip(depth / max_depth, 0.0, 1.0)
    depth = (depth * 255).astype(np.uint8)
    return np.repeat(depth[:, :, None], 3, axis=2)


def set_start(sim, seed=7):
    sim.seed(seed)
    agent = sim.initialize_agent(0)

    if sim.pathfinder.is_loaded:
        sim.pathfinder.seed(seed)
        start = sim.pathfinder.get_random_navigable_point()
    else:
        print("[WARN] pathfinder not loaded; using origin.")
        start = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    state = habitat_sim.AgentState()
    state.position = start
    state.rotation = utils.quat_from_magnum(
        mn.Quaternion.rotation(mn.Rad(0.0), mn.Vector3(0.0, 1.0, 0.0))
    )
    agent.set_state(state)

    print("[INFO] Agent start position:", agent.get_state().position)
    print("[INFO] Pathfinder loaded:", sim.pathfinder.is_loaded)

    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="apt_0")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    scene_handle = resolve_scene_handle(args.scene)
    print("[INFO] Loading ReplicaCAD scene:")
    print("       ", scene_handle)

    sim = make_sim(scene_handle, width=args.width, height=args.height)

    try:
        set_start(sim, seed=args.seed)

        frames = []
        depth_frames = []

        obs = sim.get_sensor_observations()
        frames.append(to_rgb(obs))
        depth_frames.append(depth_vis(obs))

        actions = []
        actions += ["turn_left"] * 24
        actions += ["move_forward"] * 12
        actions += ["turn_right"] * 12
        actions += ["move_forward"] * 12

        for i, action in enumerate(actions):
            obs = sim.step(action)
            frames.append(to_rgb(obs))
            depth_frames.append(depth_vis(obs))

            if i % 10 == 0:
                print(f"[INFO] step={i:03d}, action={action}")

        safe_scene_name = args.scene.replace("/", "_").replace("\\", "_").replace(".", "_")
        rgb_video = out_dir / f"smoke_replicacad_{safe_scene_name}_rgb.mp4"
        depth_video = out_dir / f"smoke_replicacad_{safe_scene_name}_depth.mp4"

        imageio.mimsave(rgb_video, frames, fps=10)
        imageio.mimsave(depth_video, depth_frames, fps=10)

        print("[OK] Saved RGB video:", rgb_video)
        print("[OK] Saved Depth video:", depth_video)

    finally:
        sim.close()


if __name__ == "__main__":
    main()
