import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.coco_utils import COCOAdversarialDataset, collate_fn
from models.diffusion_model import AdvCEODiffusion
from models.detector import YOLODetector
from trainers.adversarial_trainer import AdversarialTrainer
from my_visualize import plot_threeway_comparison

class FilteredCOCOAdversarialDataset(COCOAdversarialDataset):
    def __init__(self, *args, max_samples=None, filter_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_indices = []
        self.filter_classes = filter_classes
        total_len = super().__len__()
        for idx in range(total_len):
            data = super().__getitem__(idx)
            target = data[1]
            if 'boxes' in target and hasattr(target['boxes'], 'shape') and target['boxes'].shape[0] > 0:
                if self.filter_classes is None or any(label in self.filter_classes for label in target['labels'].tolist()):
                    self.valid_indices.append(idx)
            if max_samples is not None and len(self.valid_indices) >= max_samples:
                break
        print(f"Total images: {total_len}, Valid images with targets in selected classes: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        data, target = super().__getitem__(real_idx)
        if self.filter_classes is not None:
            mask = [i for i, label in enumerate(target['labels']) if label in self.filter_classes]
            target['boxes'] = target['boxes'][mask]
            target['labels'] = target['labels'][mask]
        return data, target

def get_coco_to_custom(config):
    coco_id_map = {
        "bear": 21,
        "zebra": 24,
        "car": 2,
        "horse": 17,
        "sheep": 19,
        "bus": 5,
        "cow": 20,
        "microwave": 69,
        "bottle": 39,
        "couch": 57
    }
    user_cats = config.get('categories', list(coco_id_map.keys()))
    coco_to_custom = {coco_id_map[name]: idx for idx, name in enumerate(user_cats) if name in coco_id_map}
    custom_label_ids = list(coco_to_custom.keys())
    return coco_to_custom, custom_label_ids, user_cats

def main():
    with open('configs/default.yaml', 'r', encoding='utf-8', errors='ignore') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coco_to_custom, custom_label_ids, categories = get_coco_to_custom(config)
    config['categories'] = categories

    detector = YOLODetector(
        model_name=config.get("detector", "yolov5su.pt"),
        target_classes=categories,
        coco_to_custom=coco_to_custom
    )
    model = AdvCEODiffusion(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    ann_file = config['ann_path']
    if os.path.isdir(ann_file):
        ann_file = os.path.join(ann_file, "instances_train2017.json")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
    if not os.path.isdir(config['data_path']):
        raise FileNotFoundError(f"COCO data_path not found: {config['data_path']}")

    train_dataset = FilteredCOCOAdversarialDataset(
        root=config['data_path'],
        ann_file=ann_file,
        transform=transforms.ToTensor(),
        categories=categories,
        filter_classes=custom_label_ids,
        max_samples=config.get('max_samples', None)
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    trainer = AdversarialTrainer(model, detector, optimizer, config, device=device)

    for epoch in range(config['epochs']):
        loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss:.4f}")
        try:
            sample_data = next(iter(dataloader))
            clean_img = sample_data[0][0]
            adv_img = trainer.generate_adversarial_example(clean_img.unsqueeze(0)).squeeze(0)
            patched_img = trainer.generate_patched_example(clean_img.unsqueeze(0)).squeeze(0)
            plot_threeway_comparison(
                clean_img, adv_img, patched_img, detector, categories, save_path=f"compare_epoch{epoch+1}.png"
            )
        except Exception as e:
            print(f"可视化样本失败: {e}")

if __name__ == "__main__":
    main()
