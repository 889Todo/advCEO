import torch
import random
from utils.patch_fusion import hsv_fusion
from losses.detection_loss import DetectionLoss

def to_float(val):
    return val.item() if hasattr(val, "item") else float(val)

class AdversarialTrainer:
    def __init__(self, model, detector, optimizer, config, device=None):
        self.model = model
        self.detector = detector
        self.optimizer = optimizer
        self.config = config
        self.device = device if device is not None else model.device
        self.loss_mse = torch.nn.MSELoss()
        self.loss_det = DetectionLoss(target_class_id=self._get_target_class_id()).to(self.device)


    def _get_target_class_id(self):
        return self.config['cat_id'] if 'cat_id' in self.config else 18

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0
        for images, targets in dataloader:
            images = images.to(self.device)
            for t in targets:
                t['boxes'] = t['boxes'].to(self.device)
                t['labels'] = t['labels'].to(self.device)

            noise = torch.randn_like(images)
            timesteps = torch.randint(0, self.config['num_diffusion_steps'], (images.size(0),), device=images.device)
            noisy_images = self.model.add_noise(images, noise, timesteps)
            pred_noise = self.model(noisy_images, timesteps)
            loss_mse = self.loss_mse(pred_noise, noise)
            adv_images = (images + 0.1 * pred_noise).clamp(0, 1)

            with torch.no_grad():
                clean_preds = self.detector(images)
            adv_preds = self.detector(adv_images)
            gt_boxes_batch = [t['boxes'] for t in targets]
            loss_adv = self.loss_det(adv_preds, gt_boxes_batch)

            loss = (
                self.config['loss_weights']['mse'] * loss_mse +
                self.config['loss_weights']['adv'] * loss_adv
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += to_float(loss)
            batch_count += 1

        return epoch_loss / batch_count if batch_count > 0 else 0.0

    def generate_adversarial_example(self, images, target_class=None, method='pgd'):
        """
        生成全图对抗样本
        """
        self.model.eval()
        with torch.no_grad():
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, self.config['num_diffusion_steps'], (images.size(0),), device=images.device)
            noisy_images = self.model.add_noise(images, noise, timesteps)
            pred_noise = self.model(noisy_images, timesteps)
            adv_images = (images + 0.1 * pred_noise).clamp(0, 1)
        return adv_images

    def generate_patched_example(self, images):
        """
        生成HSV融合补丁：补丁贴在检测物体表面，内容为原图与对抗噪声的HSV融合
        """
        self.model.eval()
        with torch.no_grad():
            b, c, h, w = images.shape
            patched = images.clone()
            det_results = self.detector(images)
            boxes_list = det_results['boxes']
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, self.config['num_diffusion_steps'], (b,), device=images.device)
            noisy_images = self.model.add_noise(images, noise, timesteps)
            pred_noise = self.model(noisy_images, timesteps)
            # 生成对抗补丁
            for i in range(b):
                boxes = boxes_list[i]
                if boxes.shape[0] == 0:
                    continue  # 若无目标则跳过
                idx = random.randint(0, boxes.shape[0]-1)
                x1, y1, x2, y2 = [int(x.item()) for x in boxes[idx]]
                bw, bh = x2 - x1, y2 - y1
                # 补丁为检测框中心1/3区域
                pw, ph = max(bw // 3, 8), max(bh // 3, 8)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                px1, py1 = max(cx - pw // 2, 0), max(cy - ph // 2, 0)
                px2, py2 = min(cx + pw // 2, w), min(cy + ph // 2, h)
                # 原patch和对抗噪声patch
                img_patch = images[i:i+1, :, py1:py2, px1:px2]
                noise_patch = pred_noise[i:i+1, :, py1:py2, px1:px2]
                # 对抗补丁 = img + 噪声，进行HSV融合
                adv_patch = (img_patch + 0.2 * noise_patch).clamp(0, 1)
                fused_patch = hsv_fusion(img_patch, adv_patch)
                # 贴回原图
                patched[i, :, py1:py2, px1:px2] = fused_patch[0]
            return patched
