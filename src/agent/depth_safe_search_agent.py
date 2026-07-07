from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

from src.agent.command_parser import ParsedCommand


@dataclass
class Detection:
    label: str
    bbox_xyxy: Tuple[int, int, int, int]
    score: float


class DetectorProtocol(Protocol):
    def detect(self, rgb: np.ndarray, target: str) -> Optional[Detection]:
        ...


@dataclass
class AgentOutput:
    action: str
    done: bool
    reply: str
    debug: Dict[str, Any]


class DepthSafeSearchAgent:
    """
    RGB-D visual navigation state-machine agent.

    Runtime inputs:
    - user command parsed into a target name
    - RGB observation
    - Depth observation
    - robot action/collision feedback

    It does NOT use simulator object pose, semantic scene graph,
    shortest path, or oracle top-down map.
    """

    def __init__(
        self,
        detector: Optional[DetectorProtocol] = None,
        max_steps: int = 160,
        front_obstacle_threshold_m: float = 0.55,
        stop_distance_m: float = 1.60,
        lost_target_stop_distance_m: float = 1.80,
        align_threshold: float = 0.30,
    ):
        self.detector = detector
        self.max_steps = max_steps
        self.front_obstacle_threshold_m = front_obstacle_threshold_m
        self.stop_distance_m = stop_distance_m
        self.lost_target_stop_distance_m = lost_target_stop_distance_m
        self.align_threshold = align_threshold
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.state = "IDLE"
        self.last_detection: Optional[Detection] = None
        self.last_reply = ""

        # Memory for safe stopping.
        self.ever_saw_target = False
        self.ever_approached_target = False
        self.last_seen_step = -10_000
        self.last_target_distance: Optional[float] = None
        self.last_front_depth: Optional[float] = None

    def act(self, obs: Dict[str, Any], command: ParsedCommand) -> AgentOutput:
        self.step_idx += 1

        if not command.ok or command.target is None:
            self.state = "ASK"
            return AgentOutput(
                action="stop",
                done=True,
                reply=command.message,
                debug={
                    "state": self.state,
                    "target": None,
                    "reason": "command_parse_failed",
                },
            )

        rgb = obs["rgb"]
        depth = obs["depth"]
        target = command.target

        if self.step_idx > self.max_steps:
            self.state = "FAIL"
            return AgentOutput(
                action="stop",
                done=True,
                reply=(
                    f"我已经搜索了一段时间，但还没有稳定找到 {target}。"
                    "请换一个目标或换一个起点。还需要什么？"
                ),
                debug={
                    "state": self.state,
                    "target": target,
                    "reason": "max_steps_reached",
                },
            )

        detection = self._detect(rgb, target)
        front_depth = self._front_depth(depth)
        self.last_front_depth = front_depth

        # Case 1: target visible.
        if detection is not None:
            self.ever_saw_target = True
            self.last_seen_step = self.step_idx
            self.last_detection = detection

            action, done, reply, debug = self._act_with_detection(
                detection=detection,
                depth=depth,
                target=target,
                front_depth=front_depth,
            )

            if debug.get("state") == "APPROACH":
                self.ever_approached_target = True

            if debug.get("target_distance") is not None:
                self.last_target_distance = float(debug["target_distance"])

            debug.update(
                {
                    "step_idx": self.step_idx,
                    "front_depth": front_depth,
                    "target": target,
                    "target_visible": True,
                    "bbox": detection.bbox_xyxy,
                    "score": detection.score,
                }
            )
            return AgentOutput(action=action, done=done, reply=reply, debug=debug)

        # Case 2: target is lost after we have approached it.
        # If it was recently visible and already close, assume we have arrived.
        # This prevents "走太近后丢失目标，然后 SEARCH 继续乱走".
        recently_seen = (self.step_idx - self.last_seen_step) <= 5
        close_by_last_target = (
            self.last_target_distance is not None
            and self.last_target_distance <= self.lost_target_stop_distance_m
        )
        close_by_front_depth = front_depth <= self.lost_target_stop_distance_m

        if self.ever_approached_target and recently_seen and (close_by_last_target or close_by_front_depth):
            self.state = "STOP"
            return AgentOutput(
                action="stop",
                done=True,
                reply=f"已到达 {target} 旁边，还需要什么？",
                debug={
                    "state": self.state,
                    "target": target,
                    "target_visible": False,
                    "front_depth": front_depth,
                    "last_target_distance": self.last_target_distance,
                    "reason": "lost_target_after_close_approach",
                    "action": "stop",
                    "step_idx": self.step_idx,
                },
            )

        # Case 3: target lost but we had seen it before and are close to obstacles.
        # Do not move forward blindly. Rotate to recover the target.
        if self.ever_saw_target and front_depth < 1.20:
            self.state = "RECOVER"
            return AgentOutput(
                action="turn_left",
                done=False,
                reply=f"刚刚看到过 {target}，现在目标丢失，正在原地重新寻找。",
                debug={
                    "state": self.state,
                    "target": target,
                    "target_visible": False,
                    "front_depth": front_depth,
                    "reason": "lost_target_near_obstacle_recover",
                    "action": "turn_left",
                    "step_idx": self.step_idx,
                },
            )

        # Case 4: normal search before finding target.
        action = self._search_action(front_depth)
        self.state = "SEARCH"
        return AgentOutput(
            action=action,
            done=False,
            reply=f"正在寻找 {target} ...",
            debug={
                "state": self.state,
                "target": target,
                "target_visible": False,
                "front_depth": front_depth,
                "action": action,
                "step_idx": self.step_idx,
                "reason": "normal_search",
            },
        )

    def _detect(self, rgb: np.ndarray, target: str) -> Optional[Detection]:
        if self.detector is None:
            return None
        return self.detector.detect(rgb, target)

    def _front_depth(self, depth: np.ndarray) -> float:
        h, w = depth.shape[:2]
        y1, y2 = int(0.40 * h), int(0.75 * h)
        x1, x2 = int(0.35 * w), int(0.65 * w)

        patch = depth[y1:y2, x1:x2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]

        if valid.size == 0:
            return 10.0

        return float(np.median(valid))

    def _search_action(self, front_depth: float) -> str:
        """
        Search behavior before target is found:
        - First rotate to scan the room.
        - Then move forward only if front is clearly safe.
        - If blocked, rotate.

        Important:
        If target was seen before, do not freely explore forward.
        """
        if self.ever_saw_target:
            # After seeing target, losing it should be a recovery behavior,
            # not random exploration.
            return "turn_left"

        if self.step_idx <= 24:
            return "turn_left"

        if front_depth < self.front_obstacle_threshold_m:
            return "turn_right"

        cycle = self.step_idx % 12
        if cycle in [0, 1, 2, 3]:
            return "turn_left"
        return "move_forward"

    def _act_with_detection(
        self,
        detection: Detection,
        depth: np.ndarray,
        target: str,
        front_depth: float,
    ):
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy

        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))

        if x2 <= x1 or y2 <= y1:
            self.state = "SEARCH"
            return "turn_left", False, f"正在重新寻找 {target} ...", {
                "state": self.state,
                "target": target,
                "reason": "invalid_bbox",
            }

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area_ratio = (bbox_w * bbox_h) / float(w * h)

        touches_left = x1 <= 2
        touches_right = x2 >= w - 2
        touches_top = y1 <= 2
        touches_bottom = y2 >= h - 2

        # Bottom clipping is common when we are close to furniture,
        # so it should NOT by itself prevent STOP.
        hard_clipped_bbox = touches_left or touches_right or touches_top
        bottom_clipped = touches_bottom

        cx = 0.5 * (x1 + x2)
        horizontal_offset = (cx - 0.5 * w) / (0.5 * w)

        crop_x1 = int(x1 + 0.25 * bbox_w)
        crop_x2 = int(x1 + 0.75 * bbox_w)
        crop_y1 = int(y1 + 0.20 * bbox_h)
        crop_y2 = int(y1 + 0.70 * bbox_h)

        crop_x1 = max(0, min(w - 1, crop_x1))
        crop_x2 = max(0, min(w, crop_x2))
        crop_y1 = max(0, min(h - 1, crop_y1))
        crop_y2 = max(0, min(h, crop_y2))

        patch = depth[crop_y1:crop_y2, crop_x1:crop_x2]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > 0.05) & (valid < 10.0)]

        target_distance = float(np.median(valid)) if valid.size else 10.0

        debug_base = {
            "target": target,
            "target_distance": target_distance,
            "horizontal_offset": horizontal_offset,
            "bbox_area_ratio": bbox_area_ratio,
            "hard_clipped_bbox": hard_clipped_bbox,
            "bottom_clipped": bottom_clipped,
            "score": detection.score,
        }

        bbox_too_small = bbox_area_ratio < 0.015

        # If target is centered enough and close enough, stop.
        # For "go to sofa", 1.5-1.6m is already a reasonable "beside/near" distance in demo.
        centered_enough_for_stop = abs(horizontal_offset) <= max(self.align_threshold, 0.35)

        if (
            target_distance <= self.stop_distance_m
            and centered_enough_for_stop
            and not hard_clipped_bbox
            and not bbox_too_small
        ):
            self.state = "STOP"
            return "stop", True, f"已到达 {target} 旁边，还需要什么？", {
                **debug_base,
                "state": self.state,
                "action": "stop",
                "reason": "safe_centered_close_stop",
            }

        # Align only if significantly off-center.
        if horizontal_offset < -self.align_threshold:
            self.state = "ALIGN"
            return "turn_left", False, f"看到 {target}，正在左转对齐。", {
                **debug_base,
                "state": self.state,
                "action": "turn_left",
                "reason": "target_left",
            }

        if horizontal_offset > self.align_threshold:
            self.state = "ALIGN"
            return "turn_right", False, f"看到 {target}，正在右转对齐。", {
                **debug_base,
                "state": self.state,
                "action": "turn_right",
                "reason": "target_right",
            }

        # If we are close to a detected target but not allowed to stop because of clipping,
        # do not keep moving into it. Rotate slightly to regain a cleaner view.
        if target_distance <= self.stop_distance_m and hard_clipped_bbox:
            self.state = "RECOVER"
            return "turn_left", False, f"看到 {target} 很近，但目标框贴边，正在微调视角。", {
                **debug_base,
                "state": self.state,
                "action": "turn_left",
                "reason": "close_but_hard_clipped_recover",
            }

        if front_depth < self.front_obstacle_threshold_m:
            self.state = "AVOID"
            return "turn_right", False, f"看到 {target}，但前方有障碍，正在避障。", {
                **debug_base,
                "state": self.state,
                "action": "turn_right",
                "reason": "front_obstacle",
            }

        self.state = "APPROACH"
        return "move_forward", False, f"看到 {target}，正在靠近。", {
            **debug_base,
            "state": self.state,
            "action": "move_forward",
            "reason": "approach_centered_target",
        }
