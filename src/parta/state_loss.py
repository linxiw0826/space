"""D-58 Hungarian matching and five-term A1-O object-state loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .state_head import ObjectStatePredictions


@dataclass(frozen=True)
class StateLossConfig:
    """All numerical choices are explicit; framework defaults are not used."""

    smooth_l1_beta: float = 0.1
    scene_normalization: str = "scene_diagonal"
    minimum_scene_scale_m: float = 1e-3
    existence_weight: float = 1.0
    category_weight: float = 1.0
    center_weight: float = 1.0
    extent_weight: float = 1.0
    visibility_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if self.scene_normalization != "scene_diagonal":
            raise ValueError("A1-O v1 supports only explicit scene_diagonal normalization")
        weights = (
            self.existence_weight,
            self.category_weight,
            self.center_weight,
            self.extent_weight,
            self.visibility_weight,
        )
        if weights != (1.0, 1.0, 1.0, 1.0, 1.0):
            raise ValueError("D-58 freezes all five normalized v1 weights to 1")


@dataclass
class StateTargets:
    """One scene's unordered GT set, already filtered to actual input views."""

    categories: torch.Tensor
    centers_world_m: torch.Tensor
    extents_m: torch.Tensor
    visibility: torch.Tensor
    category_valid: torch.Tensor
    center_valid: torch.Tensor
    extent_valid: torch.Tensor
    visibility_valid: torch.Tensor
    scene_scale_m: torch.Tensor
    source_dataset: str
    scene_id: str

    @property
    def num_objects(self) -> int:
        return int(self.categories.shape[0])

    def validate(self, max_frames: int) -> None:
        count = self.num_objects
        if self.centers_world_m.shape != (count, 3) or self.extents_m.shape != (count, 3):
            raise ValueError(f"{self.scene_id}: invalid center/extent target shape")
        if self.visibility.ndim != 2 or self.visibility.shape[0] != count:
            raise ValueError(f"{self.scene_id}: invalid visibility target shape")
        if self.visibility.shape[1] > max_frames:
            raise ValueError(f"{self.scene_id}: visibility exceeds model max_frames")
        if self.visibility_valid.shape != self.visibility.shape:
            raise ValueError(f"{self.scene_id}: visibility validity shape mismatch")
        for mask in (self.category_valid, self.center_valid, self.extent_valid):
            if mask.shape != (count,):
                raise ValueError(f"{self.scene_id}: invalid field mask shape")
        scale = float(self.scene_scale_m.detach().cpu())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{self.scene_id}: invalid scene scale")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    mask = mask.bool()
    if not mask.any():
        return None
    return values[mask].mean()


class ObjectStateSetLoss:
    """Compute non-differentiable assignment then differentiable set loss."""

    COMPONENTS = ("existence", "category", "center", "extent", "visibility")

    def __init__(self, config: StateLossConfig):
        self.config = config

    def _normalized_geometry(self, target: StateTargets) -> tuple[torch.Tensor, torch.Tensor]:
        scale = target.scene_scale_m.clamp_min(self.config.minimum_scene_scale_m)
        return target.centers_world_m / scale, target.extents_m / scale

    def pair_cost(self, prediction: ObjectStatePredictions, target: StateTargets) -> torch.Tensor:
        """Return valid-component-normalized ``[K,G]`` pair costs."""
        target.validate(prediction.visibility_logits.shape[-1])
        slots = prediction.existence_logits.shape[0]
        objects = target.num_objects
        if objects == 0:
            return prediction.existence_logits.new_zeros((slots, 0))
        if objects > slots:
            raise ValueError(
                f"{target.scene_id}: {objects} GT objects exceed {slots} slots; "
                "D-58 forbids silent truncation"
            )

        pair_sum = prediction.existence_logits.new_zeros((slots, objects))
        pair_count = prediction.existence_logits.new_zeros((slots, objects))

        exist = F.binary_cross_entropy_with_logits(
            prediction.existence_logits[:, None].expand(-1, objects),
            torch.ones((slots, objects), device=prediction.existence_logits.device),
            reduction="none",
        )
        pair_sum += exist
        pair_count += 1

        if target.category_valid.any():
            if (
                (target.categories[target.category_valid] < 0).any()
                or (target.categories[target.category_valid] >= prediction.category_logits.shape[-1]).any()
            ):
                raise ValueError(f"{target.scene_id}: valid category outside canonical vocabulary")
            safe_categories = torch.where(
                target.category_valid, target.categories, torch.zeros_like(target.categories)
            )
            category_cost = -F.log_softmax(prediction.category_logits, dim=-1)[
                :, safe_categories
            ]
            mask = target.category_valid[None, :].expand(slots, -1)
            pair_sum += category_cost * mask
            pair_count += mask

        normalized_center, normalized_extent = self._normalized_geometry(target)
        if target.center_valid.any():
            center_cost = F.smooth_l1_loss(
                prediction.center_world_normalized[:, None, :].expand(-1, objects, -1),
                normalized_center[None, :, :].expand(slots, -1, -1),
                beta=self.config.smooth_l1_beta,
                reduction="none",
            ).mean(-1)
            mask = target.center_valid[None, :].expand(slots, -1)
            pair_sum += center_cost * mask
            pair_count += mask
        if target.extent_valid.any():
            extent_cost = F.smooth_l1_loss(
                prediction.extent_normalized[:, None, :].expand(-1, objects, -1),
                normalized_extent[None, :, :].expand(slots, -1, -1),
                beta=self.config.smooth_l1_beta,
                reduction="none",
            ).mean(-1)
            mask = target.extent_valid[None, :].expand(slots, -1)
            pair_sum += extent_cost * mask
            pair_count += mask

        frame_count = target.visibility.shape[1]
        if target.visibility_valid.any():
            logits = prediction.visibility_logits[:, None, :frame_count].expand(-1, objects, -1)
            labels = target.visibility[None, :, :].expand(slots, -1, -1)
            raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            valid = target.visibility_valid[None, :, :].expand(slots, -1, -1)
            visibility_cost = (raw * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            object_mask = target.visibility_valid.any(-1)[None, :].expand(slots, -1)
            pair_sum += visibility_cost * object_mask
            pair_count += object_mask

        if (pair_count == 0).any():
            raise ValueError("pair with no valid D-58 matching component")
        return pair_sum / pair_count

    @staticmethod
    def assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hungarian assignment is detached and never part of autograd."""
        if cost.shape[1] == 0:
            empty = torch.empty(0, dtype=torch.long, device=cost.device)
            return empty, empty
        rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
        return (
            torch.as_tensor(rows, dtype=torch.long, device=cost.device),
            torch.as_tensor(cols, dtype=torch.long, device=cost.device),
        )

    def _scene_loss(
        self, prediction: ObjectStatePredictions, target: StateTargets
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, bool]]:
        cost = self.pair_cost(prediction, target)
        pred_indices, target_indices = self.assignment(cost)
        slot_count = prediction.existence_logits.shape[0]
        matched = torch.zeros(slot_count, dtype=torch.bool, device=cost.device)
        matched[pred_indices] = True

        if target.num_objects == 0:
            existence = F.binary_cross_entropy_with_logits(
                prediction.existence_logits, torch.zeros_like(prediction.existence_logits)
            )
        else:
            positive = F.binary_cross_entropy_with_logits(
                prediction.existence_logits[matched],
                torch.ones_like(prediction.existence_logits[matched]),
            )
            if (~matched).any():
                negative = F.binary_cross_entropy_with_logits(
                    prediction.existence_logits[~matched],
                    torch.zeros_like(prediction.existence_logits[~matched]),
                )
                existence = 0.5 * positive + 0.5 * negative
            else:
                existence = positive

        zero = prediction.existence_logits.sum() * 0.0
        losses = {"existence": existence, "category": zero, "center": zero, "extent": zero, "visibility": zero}
        active = {"existence": True, "category": False, "center": False, "extent": False, "visibility": False}
        metrics = {"center_mae_m": zero, "extent_mae_m": zero}
        if target.num_objects == 0:
            return losses, metrics, active

        category_mask = target.category_valid[target_indices]
        if category_mask.any():
            active["category"] = True
            losses["category"] = F.cross_entropy(
                prediction.category_logits[pred_indices[category_mask]],
                target.categories[target_indices[category_mask]],
            )

        normalized_center, normalized_extent = self._normalized_geometry(target)
        center_mask = target.center_valid[target_indices]
        if center_mask.any():
            active["center"] = True
            pred = prediction.center_world_normalized[pred_indices[center_mask]]
            gt = normalized_center[target_indices[center_mask]]
            losses["center"] = F.smooth_l1_loss(
                pred, gt, beta=self.config.smooth_l1_beta
            )
            metrics["center_mae_m"] = (
                (pred * target.scene_scale_m - target.centers_world_m[target_indices[center_mask]])
                .abs()
                .mean()
                .detach()
            )
        extent_mask = target.extent_valid[target_indices]
        if extent_mask.any():
            active["extent"] = True
            pred = prediction.extent_normalized[pred_indices[extent_mask]]
            gt = normalized_extent[target_indices[extent_mask]]
            losses["extent"] = F.smooth_l1_loss(
                pred, gt, beta=self.config.smooth_l1_beta
            )
            metrics["extent_mae_m"] = (
                (pred * target.scene_scale_m - target.extents_m[target_indices[extent_mask]])
                .abs()
                .mean()
                .detach()
            )

        frame_count = target.visibility.shape[1]
        visibility_logits = prediction.visibility_logits[pred_indices, :frame_count]
        visibility_gt = target.visibility[target_indices]
        visibility_mask = target.visibility_valid[target_indices]
        raw_visibility = F.binary_cross_entropy_with_logits(
            visibility_logits, visibility_gt, reduction="none"
        )
        masked_visibility = _masked_mean(raw_visibility, visibility_mask)
        if masked_visibility is not None:
            active["visibility"] = True
            losses["visibility"] = masked_visibility
        return losses, metrics, active

    def __call__(
        self,
        predictions: ObjectStatePredictions,
        targets: Sequence[StateTargets],
    ) -> dict[str, torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size = predictions.existence_logits.shape[0]
        if len(targets) != batch_size:
            raise ValueError("prediction/target batch size mismatch")
        scene_losses = []
        scene_metrics = []
        scene_active = []
        assignments = []
        for index, target in enumerate(targets):
            prediction = ObjectStatePredictions(
                existence_logits=predictions.existence_logits[index],
                category_logits=predictions.category_logits[index],
                center_world_normalized=predictions.center_world_normalized[index],
                extent_normalized=predictions.extent_normalized[index],
                visibility_logits=predictions.visibility_logits[index],
                slots=predictions.slots[index],
            )
            cost = self.pair_cost(prediction, target)
            assignments.append(self.assignment(cost))
            losses, metrics, active = self._scene_loss(prediction, target)
            scene_losses.append(losses)
            scene_metrics.append(metrics)
            scene_active.append(active)

        # Valid element -> scene mean happened above. Here scenes are averaged
        # within source, then sources receive equal weight.
        sources = sorted(set(target.source_dataset for target in targets))
        reduced = {}
        for component in self.COMPONENTS:
            source_means = []
            for source in sources:
                indices = [
                    i
                    for i, target in enumerate(targets)
                    if target.source_dataset == source and scene_active[i][component]
                ]
                if indices:
                    source_means.append(torch.stack([scene_losses[i][component] for i in indices]).mean())
            reduced[component] = (
                torch.stack(source_means).mean()
                if source_means
                else predictions.existence_logits.sum() * 0.0
            )
        total = sum(reduced.values())
        result: dict[str, torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]]] = {
            "loss_state": total,
            **{f"loss_{key}": value for key, value in reduced.items()},
            "assignments": assignments,
        }
        for metric in ("center_mae_m", "extent_mae_m"):
            result[metric] = torch.stack([item[metric] for item in scene_metrics]).mean()
        return result
