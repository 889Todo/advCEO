import torch
from tqdm import tqdm
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
        for images, targets in tqdm(dataloader, desc="Training"):
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
            tqdm.write(
                f"Batch Loss: {to_float(loss):.4f} | MSE: {to_float(loss_mse):.4f} | Adv: {to_float(loss_adv):.4f}"
            )

        return epoch_loss / batch_count if batch_count > 0 else 0.0