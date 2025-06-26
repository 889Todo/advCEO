import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name='yolov5su', target_classes=
    ["bear", "zebra", "car", "horse", "sheep", "bus", "cow", "microwave", "bottle", "couch"], coco_to_custom=None):
        """
        model_name: 检测器模型名（如'yolov5su', 'yolov8s', 'yolo11n'）
        target_classes: 训练/测试使用的类别名（如['person', ...]）
        coco_to_custom: COCO类别id到自定义类别id的映射（如{0:0, 2:1, ...}）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name).to(self.device)
        self.target_classes = target_classes
        self.coco_to_custom = coco_to_custom

    def filter_and_remap(self, boxes, scores, labels):
        """仅保留目标类别，并做类别id remap"""
        if self.coco_to_custom is None:
            return boxes, scores, labels
        keep = []
        remapped = []
        for i, c in enumerate(labels.tolist()):
            if c in self.coco_to_custom:
                keep.append(i)
                remapped.append(self.coco_to_custom[c])
        if len(keep) == 0:
            return (torch.empty((0,4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long))
        return boxes[keep], scores[keep], torch.tensor(remapped, dtype=torch.long)

    def __call__(self, images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("input images must be torch.Tensor")
        if images.dtype != torch.float32:
            images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        images = images.to(self.device)
        results = self.model(images)
        boxes_list, scores_list, labels_list = [], [], []
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu()
                scores = r.boxes.conf.cpu()
                labels = r.boxes.cls.cpu().long()
                boxes, scores, labels = self.filter_and_remap(boxes, scores, labels)
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
            else:
                boxes_list.append(torch.empty((0,4)))
                scores_list.append(torch.empty((0,)))
                labels_list.append(torch.empty((0,), dtype=torch.long))
        return {
            'boxes': boxes_list,
            'scores': scores_list,
            'labels': labels_list
        }
