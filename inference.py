"""Vision Mamba 推理与可视化脚本

功能包括：
    1. 加载训练好的模型进行推理
    2. 可视化预测结果
    3. 混淆矩阵分析
    4. 错误样本分析
    5. 注意力图可视化（patch重要性）
    6. 模型性能统计

使用方法:
    python inference.py --model_path ./checkpoints/best_model.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

from vision_mamba import create_vision_mamba_mnist, VisionMamba

# 解决中文显示问题
def set_chinese_font():
    # 动态选择系统可用字体，避免硬编码不可用的字体名
    from matplotlib import font_manager
    preferred = [
        'Noto Sans CJK SC', 'Noto Serif CJK SC',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for p in preferred:
        if p in available:
            plt.rcParams['font.sans-serif'] = [p]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"已设置中文字体：{p}")
            return p
    # 回退
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("未能找到首选中文字体，已回退到 DejaVu Sans")
    return 'DejaVu Sans'


class ModelInference:
    """模型推理类：封装推理和分析功能"""

    def __init__(
            self,
            model: VisionMamba,
            device: torch.device,
            class_names: List[str] = None
    ):
        """初始化推理器

        Args:
            model: 训练好的Vision Mamba模型
            device: 推理设备
            class_names: 类别名称列表
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_names = class_names or [str(i) for i in range(10)]

    @torch.no_grad()
    def predict_single(self, image: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """预测单张图像

        Args:
            image: 输入图像 shape (1, 28, 28) 或 (28, 28)

        Returns:
            pred_class: 预测类别
            probs: 类别概率分布 shape (10,)
        """
        # 确保输入维度正确
        if image.dim() == 2:
            image = image.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, 28, 28) -> (1, 1, 28, 28)

        image = image.to(self.device)

        # 前向传播
        logits = self.model(image)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()

        return pred_class, probs.squeeze()

    @torch.no_grad()
    def predict_batch(
            self,
            data_loader: DataLoader,
            max_samples: int = None
    ) -> Dict[str, np.ndarray]:
        """批量预测并收集结果

        Args:
            data_loader: 数据加载器
            max_samples: 最大预测样本数（None表示全部）

        Returns:
            results: 包含预测结果的字典
        """
        all_preds = []
        all_labels = []
        all_probs = []
        all_images = []

        total_samples = 0

        for images, labels in tqdm(data_loader, desc="推理中"):
            if max_samples and total_samples >= max_samples:
                break

            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            all_images.append(images.cpu().numpy())

            total_samples += images.size(0)

        results = {
            'predictions': np.concatenate(all_preds),
            'labels': np.concatenate(all_labels),
            'probabilities': np.concatenate(all_probs),
            'images': np.concatenate(all_images)
        }

        return results

    @torch.no_grad()
    def get_patch_attention(self, image: torch.Tensor) -> np.ndarray:
        """获取patch级别的注意力权重（通过最后一层的激活值近似）

        Args:
            image: 输入图像 shape (1, 1, 28, 28)

        Returns:
            attention_map: 注意力图 shape (7, 7) for patch_size=4
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # 前向传播到归一化层之前
        x = self.model.patch_embed(image)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for layer in self.model.layers:
            x = layer(x)

        # 计算每个patch的L2范数作为重要性度量
        attention = torch.norm(x, dim=-1).squeeze()  # shape (n_patches,)
        attention = attention.cpu().numpy()

        # 重塑为2D网格
        n_patches_per_side = int(np.sqrt(len(attention)))
        attention_map = attention.reshape(n_patches_per_side, n_patches_per_side)

        # 归一化到[0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        return attention_map


class Visualizer:
    """可视化类：封装各种可视化功能"""

    def __init__(self, save_dir: str = "./visualizations"):
        """初始化可视化器

        Args:
            save_dir: 图像保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        set_chinese_font()

    def plot_predictions(
            self,
            images: np.ndarray,
            predictions: np.ndarray,
            labels: np.ndarray,
            probabilities: np.ndarray,
            num_samples: int = 16,
            save_name: str = "predictions.png"
    ):
        """可视化预测结果（网格布局）

        Args:
            images: 图像数组 shape (N, 1, 28, 28)
            predictions: 预测结果 shape (N,)
            labels: 真实标签 shape (N,)
            probabilities: 预测概率 shape (N, 10)
            num_samples: 显示样本数
            save_name: 保存文件名
        """
        num_samples = min(num_samples, len(images))
        cols = 4
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = axes.flatten()

        for idx in range(num_samples):
            ax = axes[idx]

            # 显示图像
            img = images[idx].squeeze()
            ax.imshow(img, cmap='gray')

            # 获取预测信息
            pred = predictions[idx]
            true = labels[idx]
            prob = probabilities[idx][pred]

            # 设置标题（正确为绿色，错误为红色）
            is_correct = pred == true
            color = 'green' if is_correct else 'red'
            title = f"预测:{pred} (真实:{true})\n置信度:{prob:.2%}"
            ax.set_title(title, color=color, fontsize=10, fontweight='bold')

            ax.axis('off')

        # 隐藏多余的子图
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测结果已保存: {save_path}")

    def plot_confusion_matrix(
            self,
            predictions: np.ndarray,
            labels: np.ndarray,
            class_names: List[str],
            save_name: str = "confusion_matrix.png"
    ):
        """绘制混淆矩阵

        Args:
            predictions: 预测结果
            labels: 真实标签
            class_names: 类别名称
            save_name: 保存文件名
        """
        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)

        # 绘制热图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': '样本数量'}
        )

        ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
        ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
        ax.set_title('混淆矩阵', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 混淆矩阵已保存: {save_path}")

    def plot_error_analysis(
            self,
            images: np.ndarray,
            predictions: np.ndarray,
            labels: np.ndarray,
            probabilities: np.ndarray,
            num_errors: int = 20,
            save_name: str = "error_analysis.png"
    ):
        """分析并可视化错误样本

        Args:
            images: 图像数组
            predictions: 预测结果
            labels: 真实标签
            probabilities: 预测概率
            num_errors: 显示的错误样本数
            save_name: 保存文件名
        """
        # 找出所有错误样本
        error_mask = predictions != labels
        error_indices = np.where(error_mask)[0]

        if len(error_indices) == 0:
            print("🎉 没有发现错误样本！模型表现完美！")
            return

        # 按置信度排序（高置信度错误更值得关注）
        error_confidences = probabilities[error_indices, predictions[error_indices]]
        sorted_indices = error_indices[np.argsort(error_confidences)[::-1]]

        num_errors = min(num_errors, len(sorted_indices))
        cols = 5
        rows = (num_errors + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        for i, idx in enumerate(sorted_indices[:num_errors]):
            ax = axes[i]

            img = images[idx].squeeze()
            pred = predictions[idx]
            true = labels[idx]
            conf = probabilities[idx][pred]

            ax.imshow(img, cmap='gray')
            ax.set_title(
                f"预测:{pred}→真实:{true}\n置信度:{conf:.2%}",
                color='red',
                fontsize=9,
                fontweight='bold'
            )
            ax.axis('off')

        # 隐藏多余的子图
        for i in range(num_errors, len(axes)):
            axes[i].axis('off')

        plt.suptitle(
            f'错误样本分析 (共{len(error_indices)}个错误)',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 错误分析已保存: {save_path}")
        print(f"   错误率: {len(error_indices) / len(predictions) * 100:.2f}%")

    def plot_probability_distribution(
            self,
            probabilities: np.ndarray,
            predictions: np.ndarray,
            labels: np.ndarray,
            save_name: str = "probability_distribution.png"
    ):
        """绘制预测概率分布

        Args:
            probabilities: 预测概率
            predictions: 预测结果
            labels: 真实标签
            save_name: 保存文件名
        """
        # 获取预测类别的置信度
        confidences = probabilities[np.arange(len(predictions)), predictions]

        # 区分正确和错误的预测
        correct_mask = predictions == labels
        correct_conf = confidences[correct_mask]
        wrong_conf = confidences[~correct_mask]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：直方图对比
        ax = axes[0]
        bins = np.linspace(0, 1, 30)
        ax.hist(correct_conf, bins=bins, alpha=0.7, label='正确预测', color='green', edgecolor='black')
        ax.hist(wrong_conf, bins=bins, alpha=0.7, label='错误预测', color='red', edgecolor='black')
        ax.set_xlabel('预测置信度', fontsize=12, fontweight='bold')
        ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
        ax.set_title('预测置信度分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 右图：统计信息
        ax = axes[1]
        ax.axis('off')

        stats_text = f"""
        📊 统计信息
        {'=' * 40}

        总样本数: {len(predictions):,}
        正确预测: {np.sum(correct_mask):,}
        错误预测: {np.sum(~correct_mask):,}
        准确率: {np.mean(correct_mask) * 100:.2f}%

        置信度统计:
        ─────────────────────────
        正确预测平均置信度: {np.mean(correct_conf):.4f}
        错误预测平均置信度: {np.mean(wrong_conf) if len(wrong_conf) > 0 else 0:.4f}

        高置信度正确 (>0.9): {np.sum(correct_conf > 0.9):,}
        高置信度错误 (>0.9): {np.sum(wrong_conf > 0.9) if len(wrong_conf) > 0 else 0:,}
        低置信度正确 (<0.5): {np.sum(correct_conf < 0.5):,}
        """

        ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 概率分布已保存: {save_path}")

    def plot_attention_maps(
            self,
            images: np.ndarray,
            attention_maps: List[np.ndarray],
            predictions: np.ndarray,
            labels: np.ndarray,
            num_samples: int = 8,
            save_name: str = "attention_maps.png"
    ):
        """可视化注意力图（patch重要性）

        Args:
            images: 原始图像
            attention_maps: 注意力图列表
            predictions: 预测结果
            labels: 真实标签
            num_samples: 显示样本数
            save_name: 保存文件名
        """
        num_samples = min(num_samples, len(images))

        fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # 原始图像
            ax = axes[i, 0]
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(f'原图\n真实:{labels[i]}', fontsize=10)
            ax.axis('off')

            # 注意力图
            ax = axes[i, 1]
            im = ax.imshow(attention_maps[i], cmap='hot', interpolation='nearest')
            ax.set_title('Patch重要性', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 叠加图
            ax = axes[i, 2]
            # 将注意力图上采样到原图大小
            from scipy.ndimage import zoom
            attention_upsampled = zoom(attention_maps[i], 28 / attention_maps[i].shape[0], order=1)
            ax.imshow(images[i].squeeze(), cmap='gray', alpha=0.6)
            ax.imshow(attention_upsampled, cmap='hot', alpha=0.4)
            ax.set_title(f'叠加图\n预测:{predictions[i]}', fontsize=10)
            ax.axis('off')

        plt.suptitle('Patch注意力可视化', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 注意力图已保存: {save_path}")

    def plot_per_class_accuracy(
            self,
            predictions: np.ndarray,
            labels: np.ndarray,
            class_names: List[str],
            save_name: str = "per_class_accuracy.png"
    ):
        """绘制每个类别的准确率

        Args:
            predictions: 预测结果
            labels: 真实标签
            class_names: 类别名称
            save_name: 保存文件名
        """
        num_classes = len(class_names)
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)

        for label, pred in zip(labels, predictions):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

        class_accuracy = class_correct / (class_total + 1e-8) * 100

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(class_names, class_accuracy, color='steelblue', edgecolor='black', linewidth=1.5)

        # 在柱子上标注数值
        for i, (bar, acc, total) in enumerate(zip(bars, class_accuracy, class_total)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{acc:.1f}%\n({int(class_correct[i])}/{int(total)})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 添加平均线
        avg_acc = np.mean(class_accuracy)
        ax.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2, label=f'平均准确率: {avg_acc:.2f}%')

        ax.set_xlabel('类别', fontsize=12, fontweight='bold')
        ax.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
        ax.set_title('各类别准确率分析', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim([0, 105])
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 各类别准确率已保存: {save_path}")


def load_model(model_path: str, device: torch.device) -> VisionMamba:
    """加载训练好的模型

    Args:
        model_path: 模型检查点路径
        device: 设备

    Returns:
        model: 加载好的模型
    """
    print(f"📂 加载模型: {model_path}")

    # 创建模型（使用默认配置）
    model = create_vision_mamba_mnist()
    model = torch.compile(model)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 模型加载成功 (Epoch: {checkpoint.get('epoch', 'Unknown')})")
        if 'best_val_acc' in checkpoint:
            print(f"   最佳验证准确率: {checkpoint['best_val_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ 模型加载成功")

    return model


def get_test_loader(batch_size: int = 128, data_dir: str = "./data") -> DataLoader:
    """创建测试数据加载器

    Args:
        batch_size: 批次大小
        data_dir: 数据集目录

    Returns:
        test_loader: 测试数据加载器
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_loader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Vision Mamba推理与可视化')

    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集目录')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                        help='可视化结果保存目录')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大测试样本数（None表示全部）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备')
    parser.add_argument('--show_attention', action='store_true',
                        help='是否显示注意力图（较慢）')

    return parser.parse_args()


def main():
    """主函数：完整的推理和可视化流程"""
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}\n")

    # 加载模型
    model = load_model(args.model_path, device)

    # 创建推理器
    inference = ModelInference(model, device)

    # 创建可视化器
    visualizer = Visualizer(args.save_dir)

    # 加载测试数据
    print("\n📦 加载测试数据...")
    test_loader = get_test_loader(args.batch_size, args.data_dir)

    # 批量推理
    print("\n🔮 开始推理...")
    results = inference.predict_batch(test_loader, args.max_samples)

    images = results['images']
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']

    # 计算总体准确率
    accuracy = np.mean(predictions == labels) * 100
    print(f"\n{'=' * 60}")
    print(f"📊 总体准确率: {accuracy:.2f}%")
    print(f"{'=' * 60}\n")

    # 打印分类报告
    print("📋 详细分类报告:")
    print(classification_report(labels, predictions, target_names=[str(i) for i in range(10)]))

    # 开始可视化
    print("\n🎨 生成可视化...")

    # 1. 预测结果展示
    print("  1/6 绘制预测结果...")
    visualizer.plot_predictions(images, predictions, labels, probabilities, num_samples=16)

    # 2. 混淆矩阵
    print("  2/6 绘制混淆矩阵...")
    visualizer.plot_confusion_matrix(predictions, labels, [str(i) for i in range(10)])

    # 3. 错误分析
    print("  3/6 分析错误样本...")
    visualizer.plot_error_analysis(images, predictions, labels, probabilities, num_errors=20)

    # 4. 概率分布
    print("  4/6 绘制概率分布...")
    visualizer.plot_probability_distribution(probabilities, predictions, labels)

    # 5. 各类别准确率
    print("  5/6 绘制各类别准确率...")
    visualizer.plot_per_class_accuracy(predictions, labels, [str(i) for i in range(10)])

    # 6. 注意力图（可选，比较慢）
    if args.show_attention:
        print("  6/6 生成注意力图（较慢）...")
        attention_maps = []
        num_attention_samples = min(8, len(images))

        for i in tqdm(range(num_attention_samples), desc="计算注意力"):
            img_tensor = torch.from_numpy(images[i:i + 1])
            attention = inference.get_patch_attention(img_tensor)
            attention_maps.append(attention)

        visualizer.plot_attention_maps(
            images[:num_attention_samples],
            attention_maps,
            predictions[:num_attention_samples],
            labels[:num_attention_samples]
        )
    else:
        print("  6/6 跳过注意力图（使用 --show_attention 启用）")

    print(f"\n{'=' * 60}")
    print(f"✨ 所有可视化已完成！")
    print(f"📁 结果保存在: {args.save_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()