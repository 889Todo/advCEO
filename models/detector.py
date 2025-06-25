import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name='yolov5su'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name).to(self.device)

    def __call__(self, images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("Input images must be a torch.Tensor")
        # 确保输入为 float32 且像素归一化到[0,1]
        if images.dtype != torch.float32:
            images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        images = images.to(self.device)
        results = self.model(images)
        boxes_list, scores_list, labels_list = [], [], []
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                boxes_list.append(r.boxes.xyxy.cpu())
                scores_list.append(r.boxes.conf.cpu())
                labels_list.append(r.boxes.cls.cpu().long())
            else:
                boxes_list.append(torch.empty((0,4)))
                scores_list.append(torch.empty((0,)))
                labels_list.append(torch.empty((0,), dtype=torch.long))
        return {
            'boxes': boxes_list,
            'scores': scores_list,
            'labels': labels_list
        }