"""
Habitat / ReplicaCAD backend for the embodied navigation demo.

Design goal:
- Wrap the already-working smoke-test logic into a reusable class.
- Keep sim_cfg.gpu_device_id = -1 for Windows 10 + WSL2 compatibility.
- Provide reset / step / get_observation / save_video / close.
- Expose only RGB, Depth, robot state, and collision/action result to the Agent.

Important:
- Do NOT use semantic scene, object pose, shortest path, or oracle top-down map
  inside the Agent policy.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np

# Keep Habitat logs quieter.
# These must be set before importing habitat_sim.
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat_sim
import magnum as mn
from habitat_sim.utils import common as utils


VALID_ACTIONS = {"move_forward", "turn_left", "turn_right", "stop"}


@dataclass
class HabitatBackendConfig:
    data_dir: str = "data"
    scene: str = "apt_0"
    width: int = 320
    height: int = 240
    sensor_height: float = 1.25
    move_forward_amount: float = 0.25
    turn_angle: float = 15.0
    enable_physics: bool = True

    # Critical for your current Windows 10 + WSL2 setup.
    # Do NOT change this to 0.
    gpu_device_id: int = -1


class HabitatBackend:
    def __init__(self, config: HabitatBackendConfig):
        self.config = config
        self.sim: Optional[habitat_sim.Simulator] = None
        self.agent: Optional[habitat_sim.Agent] = None
        self.last_obs: Optional[Dict[str, Any]] = None
        self.rgb_frames: List[np.ndarray] = []
        self.depth_frames: List[np.ndarray] = []
        self.step_count: int = 0
        self.scene_handle: Optional[str] = None

    # -------------------------
    # Config helpers
    # -------------------------
    def _dataset_config_path(self) -> Path:
        return Path(self.config.data_dir) / "replica_cad" / "replicaCAD.scene_dataset_config.json"

    def _make_camera_sensor(self, uuid: str, sensor_type: habitat_sim.SensorType):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [self.config.height, self.config.width]
        spec.position = [0.0, self.config.sensor_height, 0.0]
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        return spec

    def _make_sim_config(self, scene_handle: str) -> habitat_sim.Configuration:
        dataset_cfg = self._dataset_config_path()
        if not dataset_cfg.exists():
            raise FileNotFoundError(
                f"ReplicaCAD config not found: {dataset_cfg}\n"
                "Expected: data/replica_cad/replicaCAD.scene_dataset_config.json"
            )

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = str(dataset_cfg)
        sim_cfg.scene_id = scene_handle
        sim_cfg.enable_physics = self.config.enable_physics

        # Critical WSL2 fix from your handoff document.
        sim_cfg.gpu_device_id = self.config.gpu_device_id

        rgb_sensor = self._make_camera_sensor("rgb", habitat_sim.SensorType.COLOR)
        depth_sensor = self._make_camera_sensor("depth", habitat_sim.SensorType.DEPTH)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=self.config.move_forward_amount),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=self.config.turn_angle),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=self.config.turn_angle),
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def list_scenes(self) -> List[str]:
        """
        List ReplicaCAD scene handles.

        This creates a temporary lightweight simulator with scene_id='NONE'.
        It is only for configuration / debugging, not used by the Agent.
        """
        dataset_cfg = self._dataset_config_path()
        if not dataset_cfg.exists():
            raise FileNotFoundError(f"Missing ReplicaCAD config: {dataset_cfg}")

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = str(dataset_cfg)
        sim_cfg.scene_id = "NONE"
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = self.config.gpu_device_id

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        tmp = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        try:
            handles = list(tmp.metadata_mediator.get_scene_handles())
            return [h for h in handles if h and h != "NONE"]
        finally:
            tmp.close()

    def resolve_scene(self, scene: str) -> str:
        """
        Accepts 'apt_0' or any available full scene handle.
        If exact match fails, tries substring match.
        """
        handles = self.list_scenes()
        if scene in handles:
            return scene

        for h in handles:
            if scene in h:
                return h

        if not handles:
            raise RuntimeError("No ReplicaCAD scene handles found.")

        available = "\n".join(f"  - {h}" for h in handles[:20])
        raise ValueError(
            f"Scene '{scene}' not found.\n"
            f"First available scenes:\n{available}"
        )

    # -------------------------
    # Public API
    # -------------------------
    def reset(
        self,
        scene: Optional[str] = None,
        seed: int = 7,
        start_position: Optional[np.ndarray] = None,
        start_yaw: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Reset simulator and return the first observation.

        start_position and start_yaw are useful for reproducible demo episodes.
        If start_position is None, a random navigable point is sampled.
        Sampling the reset point is allowed as environment initialization;
        it is not given to the Agent as target/object privileged information.
        """
        self.close()

        if scene is not None:
            self.config.scene = scene

        self.scene_handle = self.resolve_scene(self.config.scene)
        cfg = self._make_sim_config(self.scene_handle)
        self.sim = habitat_sim.Simulator(cfg)
        self.sim.seed(seed)

        if self.sim.pathfinder.is_loaded:
            self.sim.pathfinder.seed(seed)

        self.agent = self.sim.initialize_agent(0)

        rng = random.Random(seed)

        if start_position is None:
            if self.sim.pathfinder.is_loaded:
                start_position = np.array(
                    self.sim.pathfinder.get_random_navigable_point(),
                    dtype=np.float32,
                )
            else:
                start_position = np.zeros(3, dtype=np.float32)
        else:
            start_position = np.asarray(start_position, dtype=np.float32)

        if start_yaw is None:
            start_yaw = rng.uniform(-np.pi, np.pi)

        state = habitat_sim.AgentState()
        state.position = start_position
        state.rotation = utils.quat_from_magnum(
            mn.Quaternion.rotation(
                mn.Rad(float(start_yaw)),
                mn.Vector3(0.0, 1.0, 0.0),
            )
        )
        self.agent.set_state(state)

        self.step_count = 0
        self.rgb_frames = []
        self.depth_frames = []

        obs = self.sim.get_sensor_observations()
        packed = self._pack_observation(obs, action="reset")
        self.last_obs = packed
        self._record_frames(packed)
        return packed

    def step(self, action: str) -> Dict[str, Any]:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}. Valid actions: {sorted(VALID_ACTIONS)}")
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized. Call reset() first.")

        if action == "stop":
            raw_obs = self.sim.get_sensor_observations()
        else:
            raw_obs = self.sim.step(action)

        self.step_count += 1
        packed = self._pack_observation(raw_obs, action=action)
        self.last_obs = packed
        self._record_frames(packed)
        return packed

    def get_observation(self) -> Dict[str, Any]:
        if self.last_obs is None:
            raise RuntimeError("No observation yet. Call reset() first.")
        return self.last_obs

    def save_rgb_video(self, path: str, fps: int = 10) -> str:
        if not self.rgb_frames:
            raise RuntimeError("No RGB frames recorded.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, self.rgb_frames, fps=fps)
        return str(out)

    def save_depth_video(self, path: str, fps: int = 10) -> str:
        if not self.depth_frames:
            raise RuntimeError("No depth frames recorded.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, self.depth_frames, fps=fps)
        return str(out)

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
        self.sim = None
        self.agent = None
        self.last_obs = None

    # -------------------------
    # Observation packing
    # -------------------------
    def _pack_observation(self, raw_obs: Dict[str, Any], action: str) -> Dict[str, Any]:
        rgb = raw_obs["rgb"]
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        rgb = rgb.astype(np.uint8)

        depth = raw_obs["depth"].astype(np.float32)

        agent_state = self.agent.get_state() if self.agent is not None else None
        if agent_state is not None:
            position = np.asarray(agent_state.position, dtype=np.float32).tolist()
            rotation_repr = str(agent_state.rotation)
        else:
            position = None
            rotation_repr = None

        collided = False
        if self.sim is not None:
            collided = bool(getattr(self.sim, "previous_step_collided", False))

        return {
            "rgb": rgb,
            "depth": depth,
            "agent_position": position,
            "agent_rotation": rotation_repr,
            "collided": collided,
            "last_action": action,
            "step_count": self.step_count,
            "scene": self.scene_handle,
            "width": self.config.width,
            "height": self.config.height,
        }

    def _record_frames(self, obs: Dict[str, Any]) -> None:
        self.rgb_frames.append(obs["rgb"].copy())
        self.depth_frames.append(self.depth_to_rgb(obs["depth"]))

    @staticmethod
    def depth_to_rgb(depth: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth)
        depth_vis = np.zeros_like(depth, dtype=np.float32)
        depth_vis[valid] = np.clip(depth[valid] / max_depth, 0.0, 1.0)
        depth_u8 = (depth_vis * 255.0).astype(np.uint8)
        return np.repeat(depth_u8[:, :, None], 3, axis=2)
