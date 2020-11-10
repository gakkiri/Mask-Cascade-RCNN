import torch
from torch import nn

from .fpn import build_resnet_fpn_backbone
from .rpn import RPN
from .roi import CascadeROIHeads
from utils import ShapeSpec
from structures import ImageList


class CascadeRCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
        self.rpn = RPN(cfg, self.backbone.output_shape())
        roi_head_cfg = CascadeROIHeads.from_config(cfg, self.backbone.output_shape())
        self.roi_head = CascadeROIHeads(**roi_head_cfg)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, batched_inputs):
        '''
        :param batched_input: List[Dict]
            Instance:
                filename, height, width, image(tensor), Instance(box field only), gt_classes(only 0. in this case, 1.bg)
        :return:
            Dict include all loss or detector result
        '''
        images = self.preprocess_image(batched_inputs)
        if not self.training:
            return self.inference(images)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.rpn(images, features, gt_instances)
        _, detector_losses = self.roi_head(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, images):
        features = self.backbone(images.tensor)
        proposals, _ = self.rpn(images, features, None)
        results, _ = self.roi_head(images, features, proposals, None)
        # todo: post process, now bbox is resized level.
        return results

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, 0)
        return images
