from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class RobotProfile:
    """
    Embodiment profile for the navigation demo.

    The robot is modeled as a wheeled dual-arm mobile manipulator for a navigation-only task.
    The wheeled base is controlled; the two arms are fixed and not actuated.
    """
    name: str = "WheeledDualArmNavBot"
    embodiment: str = "wheeled dual-arm mobile manipulator"
    base: str = "wheeled mobile base"
    arms: str = "two fixed arms, not actuated in the navigation task"
    sensor: str = "first-person RGB-D camera"
    actions: str = "move_forward, turn_left, turn_right, stop"
    runtime_inputs: str = "RGB, depth, robot state/action feedback, user text command"
    not_used: str = "object pose, semantic oracle, shortest path, top-down oracle map, scene graph metadata"


ROBOT_PROFILE = RobotProfile()


def robot_profile_dict() -> Dict[str, str]:
    return asdict(ROBOT_PROFILE)
