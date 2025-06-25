import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.coco_utils import COCOAdversarialDataset, collate_fn
from models.diffusion_model import AdvCEODiffusion
from models.detector import YOLODetector
from trainers.adversarial_trainer import AdversarialTrainer
from my_visualize import plot_threeway_comparison  # <--- 新增

class FilteredCOCOAdversarialDataset(COCOAdversarialDataset):
    def __init__(self, *args, max_samples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_indices = []
        total_len = super().__len__()
        for idx in range(total_len):
            data = super().__getitem__(idx)
            target = data[1]
            if 'boxes' in target and hasattr(target['boxes'], 'shape') and target['boxes'].shape[0] > 0:
                self.valid_indices.append(idx)
            if max_samples is not None and len(self.valid_indices) >= max_samples:
                break
        print(f"Total images: {total_len}, Valid images with targets: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return super().__getitem__(real_idx)

def main():
    # 加载配置
    with open('configs/default.yaml', 'r', encoding='utf-8', errors='ignore') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvCEODiffusion(config).to(device)
    detector = YOLODetector(model_name="yolov5su.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    # 检查数据和标注路径
    ann_file = config['ann_path']
    if os.path.isdir(ann_file):
        ann_file = os.path.join(ann_file, "instances_train2017.json")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
    if not os.path.isdir(config['data_path']):
        raise FileNotFoundError(f"COCO data_path not found: {config['data_path']}")

    # 用过滤后的数据集
    train_dataset = FilteredCOCOAdversarialDataset(
        root=config['data_path'],
        ann_file=ann_file,
        transform=transforms.ToTensor(),
        categories=config.get('categories', None),
        max_samples=config.get('max_samples', None)
    )


    dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        #num_workers=0,
        collate_fn=collate_fn
    )

    trainer = AdversarialTrainer(model, detector, optimizer, config, device=device)
    for epoch in range(config['epochs']):
        loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss:.4f}")

        # === 可视化三图对比示例（每个epoch展示一组样本，实际使用时请按自身数据获取方式调整） ===
        # 获取一组示例图片
        try:
            sample_data = next(iter(dataloader))
            clean_img = sample_data[0][0]  # 假设batch第一个为原图
            # 下面两步需你按实际生成方式替换
            adv_img = clean_img.clone()  # TODO: 换成真实对抗样本
            patched_img = clean_img.clone()  # TODO: 换成真实补丁图
            plot_threeway_comparison(clean_img, adv_img, patched_img, save_path=f"compare_epoch{epoch+1}.png")
        except Exception as e:
            print(f"可视化样本失败: {e}")

if __name__ == "__main__":
    main()