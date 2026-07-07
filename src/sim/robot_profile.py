from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class RobotProfile:
    """
    Navigation-task embodiment profile.

    The project focuses on navigation, so the wheeled base is controlled.
    The two arms are fixed and not actuated in this navigation-only demo.
    """
    name: str = "WheeledDualArmNavBot"
    embodiment: str = "wheeled dual-arm mobile manipulator"
    base: str = "wheeled mobile base"
    arms: str = "two fixed arms, not actuated in the navigation task"
    sensors: str = "first-person RGB-D camera"
    controlled_actions: str = "move_forward, turn_left, turn_right, stop"
    runtime_inputs: str = "RGB, depth, robot state/action feedback, user text command"
    not_used: str = "object pose, semantic oracle, shortest path, top-down oracle map, scene graph metadata"


ROBOT_PROFILE = RobotProfile()


def robot_profile_dict() -> Dict[str, str]:
    return asdict(ROBOT_PROFILE)
