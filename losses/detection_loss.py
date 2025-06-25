import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self, target_class_id=18):
        super().__init__()
        self.target_class_id = target_class_id

    def forward(self, preds, gt_boxes_batch):
        total_loss = 0.0
        eps = 1e-7
        batch_size = len(preds['boxes'])
        for boxes, scores, labels, gt_boxes in zip(preds['boxes'], preds['scores'], preds['labels'], gt_boxes_batch):
            if len(labels) == 0 or gt_boxes is None or len(gt_boxes) == 0:
                continue
            target_mask = (labels == self.target_class_id)
            if target_mask.any():
                cls_score = scores[target_mask].mean()
                cls_loss = -torch.log(cls_score + eps)
                iou = self._calculate_iou(boxes[target_mask], gt_boxes)
                iou_loss = 1 - iou.mean() if iou.numel() > 0 else 0.0
            else:
                cls_loss = 0.0
                iou_loss = 0.0
            total_loss += cls_loss + 0.5 * iou_loss
        return total_loss / batch_size

    @staticmethod
    def _calculate_iou(boxes1, boxes2):
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.tensor([])
        b1 = DetectionLoss._xywh_to_xyxy(boxes1)
        b2 = DetectionLoss._xywh_to_xyxy(boxes2)
        lt = torch.max(b1[:, None, :2], b2[:, :2])
        rb = torch.min(b1[:, None, 2:], b2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-7)
        return iou.max(dim=1)[0]

    @staticmethod
    def _xywh_to_xyxy(boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)