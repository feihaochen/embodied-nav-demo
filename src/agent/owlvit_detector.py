from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from src.agent.depth_safe_search_agent import Detection


TARGET_PROMPTS: Dict[str, List[str]] = {
    "sofa": [
        "a sofa",
        "a couch",
        "a living room sofa",
    ],
    "bed": [
        "a bed",
        "a bedroom bed",
    ],
    "table": [
        "a table",
        "a dining table",
        "a coffee table",
        "a desk",
    ],
    "chair": [
        "a chair",
        "a dining chair",
        "an armchair",
    ],
    "kitchen counter": [
        "a kitchen counter",
        "a countertop",
        "a kitchen island",
    ],
}


@dataclass
class OwlVitDetectorConfig:
    model_name: str = "google/owlvit-base-patch32"
    threshold: float = 0.07
    detect_every: int = 3
    keep_last_for: int = 2
    device: str = "auto"


class OwlVitDetector:
    """
    Text-conditioned open-vocabulary detector.

    It only uses RGB frames from Habitat.
    It does NOT use simulator object pose, semantic ids, or scene graph.
    """

    def __init__(self, config: Optional[OwlVitDetectorConfig] = None):
        self.config = config or OwlVitDetectorConfig()

        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        print(f"[OwlVitDetector] loading model={self.config.model_name} device={self.device}")
        self.processor = OwlViTProcessor.from_pretrained(self.config.model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.frame_idx = 0
        self.last_detection: Optional[Detection] = None
        self.last_detection_frame: int = -10

    def reset(self):
        self.frame_idx = 0
        self.last_detection = None
        self.last_detection_frame = -10

    def detect(self, rgb: np.ndarray, target: str) -> Optional[Detection]:
        """
        Args:
            rgb: H x W x 3 uint8 RGB image.
            target: canonical target name, e.g. sofa / bed / table / chair.

        Returns:
            Detection or None.
        """
        self.frame_idx += 1

        # To reduce compute load, run OWL-ViT every N frames.
        # For skipped frames, briefly reuse recent detection.
        if self.frame_idx % self.config.detect_every != 0:
            if (
                self.last_detection is not None
                and self.frame_idx - self.last_detection_frame <= self.config.keep_last_for
            ):
                return self.last_detection
            return None

        prompts = TARGET_PROMPTS.get(target, [f"a {target}"])
        detection = self._run_owlvit(rgb, target, prompts)

        if detection is not None:
            self.last_detection = detection
            self.last_detection_frame = self.frame_idx

        return detection

    def _run_owlvit(
        self,
        rgb: np.ndarray,
        target: str,
        prompts: List[str],
    ) -> Optional[Detection]:
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        image = Image.fromarray(rgb)
        texts = [prompts]

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # target_sizes expects (height, width)
        target_sizes = torch.tensor(
            [[image.height, image.width]],
            device=self.device,
            dtype=torch.float32,
        )

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.config.threshold,
        )[0]

        scores = results["scores"].detach().cpu()
        labels = results["labels"].detach().cpu()
        boxes = results["boxes"].detach().cpu()

        if len(scores) == 0:
            return None

        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())
        best_label_idx = int(labels[best_idx].item())
        best_prompt = prompts[best_label_idx] if best_label_idx < len(prompts) else target

        x1, y1, x2, y2 = boxes[best_idx].tolist()

        h, w = rgb.shape[:2]
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w, round(x2))))
        y2 = int(max(0, min(h, round(y2))))

        if x2 <= x1 or y2 <= y1:
            return None

        return Detection(
            label=f"{target}:{best_prompt}",
            bbox_xyxy=(x1, y1, x2, y2),
            score=best_score,
        )
