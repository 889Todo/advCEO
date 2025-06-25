import torch
import cv2
import numpy as np

def hsv_fusion(clean_img: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
    """
    将对抗补丁与原始图像在HSV空间融合，保留原始图像的亮度（Value）和饱和度（Saturation）
    输入:
        clean_img: [B, 3, H, W], 范围 [0, 1]
        patch: [B, 3, H, W], 范围 [0, 1]
    输出:
        fused_img: [B, 3, H, W]
    """
    # 转换到HSV空间
    clean_hsv = rgb_to_hsv(clean_img)
    patch_hsv = rgb_to_hsv(patch)

    # 融合策略：保留原始图像的S和V通道，仅使用补丁的H通道
    fused_hsv = torch.cat([
        patch_hsv[:, 0:1],  # 使用补丁的Hue
        clean_hsv[:, 1:3]  # 保留原图的Saturation和Value
    ], dim=1)

    # 转换回RGB空间
    return hsv_to_rgb(fused_hsv)

def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    """RGB转HSV（支持批量处理）"""
    image = image.permute(0, 2, 3, 1).cpu().numpy() * 255
    hsv_images = []
    for img in image:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[..., 0] /= 179.0  # H通道归一化
        hsv[..., 1:] /= 255.0  # S和V通道归一化
        hsv_images.append(hsv)
    return torch.from_numpy(np.stack(hsv_images)).permute(0, 3, 1, 2).to(image.device)

def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """HSV转RGB（支持批量处理）"""
    image = image.permute(0, 2, 3, 1).cpu().numpy()
    image[..., 0] *= 179.0  # 恢复H通道
    image[..., 1:] *= 255.0  # 恢复S和V通道
    rgb_images = []
    for hsv in image:
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        rgb_images.append(rgb.astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(rgb_images)).permute(0, 3, 1, 2).to(image.device)