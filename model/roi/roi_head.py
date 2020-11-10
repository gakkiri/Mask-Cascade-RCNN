# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from utils import ShapeSpec
from structures import Boxes, Instances, pairwise_iou

from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .mask_head import MaskRCNNConvUpsampleHead


def select_foreground_proposals(proposals: List[Instances], bg_label: int) -> Tuple[
    List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    def __init__(
            self,
            *,
            num_classes,
            batch_size_per_image,
            positive_sample_fraction,
            proposal_matcher,
            proposal_append_gt=True
    ):
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_sample_fraction = positive_sample_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": 512,
            "positive_sample_fraction": 0.25,
            "num_classes": 1,
            "proposal_append_gt": True,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                [0.5],
                [0, 1],
                allow_low_quality_matches=False,
            ),
        }

    def _sample_proposals(
            self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes, 1 in this case)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        # concat gt_boxes, this is a trick you can use to prevent no matched roi in early training.
        # So, it may due to zero loss in your case.
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt


class StandardROIHeads(ROIHeads):
    def __init__(
            self,
            *,
            box_in_features: List[str],
            box_pooler: ROIPooler,
            box_head: nn.Module,
            box_predictor: nn.Module,
            mask_in_features: Optional[List[str]] = None,
            mask_pooler: Optional[ROIPooler] = None,
            mask_head: Optional[nn.Module] = None,
            train_on_pred_boxes: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.train_on_pred_boxes = train_on_pred_boxes

    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        mask_in_features = ['p2', 'p3', 'p4', 'p5']
        in_channels = [input_shape[f].channels for f in mask_in_features]
        pooler_resolution = 7
        pooler_scales = list(1.0 / input_shape[k].stride for k in mask_in_features)
        sampling_ratio = 0
        pooler_type = 'ROIAlignV2'
        mask_pooler_resolution = 14
        mask_pooler = ROIPooler(
            output_size=mask_pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        shape = ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        head_cfg = MaskRCNNConvUpsampleHead.from_config(cfg, shape)
        mask_head = MaskRCNNConvUpsampleHead(**head_cfg)
        return {
            'mask_in_features': mask_in_features,
            'mask_pooler': mask_pooler,
            'mask_head': mask_head
        }

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = False
        ret.update(cls._init_box_head(cfg, input_shape))
        ret.update(cls._init_mask_head(cfg, input_shape))
        return ret


