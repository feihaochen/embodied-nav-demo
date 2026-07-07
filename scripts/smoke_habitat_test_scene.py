import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat_sim
import magnum as mn
from habitat_sim.utils import common as utils


def make_sim(scene_path: str, width: int = 640, height: int = 480):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.gpu_device_id = -1
    sim_cfg.enable_physics = False

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [height, width]
    rgb_sensor.position = [0.0, 1.25, 0.0]
    rgb_sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [height, width]
    depth_sensor.position = [0.0, 1.25, 0.0]
    depth_sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
    }

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def to_rgb(obs):
    rgb = obs["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb.astype(np.uint8)


def main():
    scene_path = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    if not Path(scene_path).exists():
        raise FileNotFoundError(
            f"Missing {scene_path}. Run datasets_download for habitat_test_scenes first."
        )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    sim = make_sim(scene_path)

    try:
        agent = sim.initialize_agent(0)

        if sim.pathfinder.is_loaded:
            start = sim.pathfinder.get_random_navigable_point()
        else:
            start = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = utils.quat_from_magnum(
            mn.Quaternion.rotation(mn.Rad(0.0), mn.Vector3(0.0, 1.0, 0.0))
        )
        agent.set_state(state)

        frames = []
        obs = sim.get_sensor_observations()
        frames.append(to_rgb(obs))

        actions = ["turn_left"] * 24 + ["move_forward"] * 12 + ["turn_right"] * 24

        for i, action in enumerate(actions):
            obs = sim.step(action)
            frames.append(to_rgb(obs))
            if i % 10 == 0:
                print(f"step={i:03d}, action={action}")

        video_path = out_dir / "smoke_habitat_test_scene.mp4"
        imageio.mimsave(video_path, frames, fps=10)
        print(f"[OK] Saved video: {video_path}")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
