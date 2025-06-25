import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

def plot_entropy(clean_img: torch.Tensor, adv_img: torch.Tensor, save_path: str = None):
    """
    绘制原始图像与对抗样本的熵对比图
    输入:
        clean_img: [3, H, W], 范围 [0, 1]
        adv_img: [3, H, W], 范围 [0, 1]
    """

    def calculate_entropy(img: torch.Tensor):
        # 计算图像局部熵
        img_np = img.permute(1, 2, 0).cpu().numpy()
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        entropy = cv2.calcHist([gray], [0], None, [256], [0, 256])
        entropy /= entropy.sum() + 1e-6
        entropy = -np.sum(entropy * np.log2(entropy + 1e-6))
        return entropy

    clean_entropy = calculate_entropy(clean_img)
    adv_entropy = calculate_entropy(adv_img)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(clean_img.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Clean Image (Entropy={clean_entropy:.2f})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(adv_img.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Adversarial Image (Entropy={adv_entropy:.2f})")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def plot_detection_results(img: torch.Tensor, pred_boxes, pred_labels, save_path: str = None):
    """
    绘制目标检测结果（边界框与类别标签）
    输入:
        img: [3, H, W], 范围 [0, 1]
        pred_boxes: [[x1,y1,x2,y2], ...] 绝对坐标
        pred_labels: 类别ID列表
    """
    img_np = img.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)

    for box, label in zip(pred_boxes, pred_labels):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor='red', linewidth=2
        ))
        plt.text(x1, y1 - 5, f"Class {label}", color='red', fontsize=12)

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def plot_threeway_comparison(clean_img: torch.Tensor, adv_img: torch.Tensor, patched_img: torch.Tensor, save_path: str = None):
    """
    三图对比：原图、对抗图、加补丁图
    输入:
        clean_img, adv_img, patched_img: [3, H, W], 范围 [0, 1]
    """
    imgs = [clean_img, adv_img, patched_img]
    titles = ["Original", "Adversarial", "Patched"]
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 3, i+1)
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()