"""Mamba Delta 可视化工具

Delta (Δ) 是 Mamba 的核心创新，控制状态空间的离散化步长。
这个脚本可视化：
    1. Delta 值的空间分布（patch-wise）
    2. Delta 值在不同层的演化
    3. Delta 值的统计特性
    4. Delta 对不同输入的响应模式
    5. Delta 与最终预测的关系

使用方法:
    python visualize_delta.py --model_path ./checkpoints/best_model.pth
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from scipy.ndimage import zoom
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

from vision_mamba import VisionMamba, create_vision_mamba_mnist

ENABLE_POS_EMBED = False  # 是否启用位置编码


def configure_fonts():
    """配置字体设置（Linux兼容）"""
    # 获取系统所有可用字体
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
    
    # Linux上常见的中文字体优先级列表
    chinese_fonts = [
        'Noto Sans CJK SC',
        'Noto Sans CJK TC', 
        'Noto Serif CJK SC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Droid Sans Fallback',
        'AR PL UMing CN',
        'AR PL UKai CN',
    ]
    
    # 备用英文字体
    fallback_fonts = ['DejaVu Sans', 'Liberation Sans', 'Arial']
    
    # 查找可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 如果没有中文字体，使用英文字体
    if selected_font is None:
        for font in fallback_fonts:
            if font in available_fonts:
                selected_font = font
                break
    
    # 设置字体
    if selected_font:
        matplotlib.rcParams['font.sans-serif'] = [selected_font]
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"✓ 字体设置成功: {selected_font}")
        return selected_font
    else:
        # 最后的备选方案
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['axes.unicode_minus'] = False
        print("⚠ 未找到合适的中文字体，使用系统默认字体")
        return 'default'


class DeltaVisualizer:
    """Delta 可视化器"""

    def __init__(self, model: VisionMamba, device: torch.device, save_dir: str = "./delta_viz"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 配置绘图样式
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")

    def extract_delta_values(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取所有层的 delta 值

        Args:
            x: 输入图像 shape (1, 1, 28, 28)

        Returns:
            delta_dict: 包含各层 delta 值的字典
        """
        deltas = {}

        with torch.no_grad():
            # Patch embedding
            x = self.model.patch_embed(x)  # (1, n_patches, d_model)
            if ENABLE_POS_EMBED:    # 添加位置编码
                x = x + self.model.pos_embed
            x = self.model.pos_drop(x)

            # 遍历每一层，提取 delta
            for layer_idx, layer in enumerate(self.model.layers):
                # 通过 norm
                x_normed = layer.norm(x)

                # 通过 in_proj
                x_and_res = layer.mixer.in_proj(x_normed)
                (x_proj, res) = x_and_res.split(
                    split_size=[layer.mixer.args.d_inner, layer.mixer.args.d_inner],
                    dim=-1
                )

                # 通过 conv1d
                x_proj = x_proj.transpose(1, 2)  # (1, d_inner, n_patches)
                x_proj = layer.mixer.conv1d(x_proj)[:, :, :x.shape[1]]
                x_proj = x_proj.transpose(1, 2)  # (1, n_patches, d_inner)
                x_proj = torch.nn.functional.silu(x_proj)

                # 提取 delta
                x_dbl = layer.mixer.x_proj(x_proj)  # (1, n_patches, dt_rank + 2*n)
                n = layer.mixer.A_log.shape[1]
                dt_rank = layer.mixer.args.dt_rank

                (delta, B, C) = x_dbl.split(split_size=[dt_rank, n, n], dim=-1)
                delta = torch.nn.functional.softplus(layer.mixer.dt_proj(delta))  # (1, n_patches, d_inner)

                # 保存 delta
                deltas[f'layer_{layer_idx}'] = delta.squeeze(0).cpu()  # (n_patches, d_inner)

                # 继续前向传播
                x = layer(x)

        return deltas

    def plot_delta_spatial_distribution(
            self,
            image: np.ndarray,
            deltas: Dict[str, torch.Tensor],
            prediction: int,
            true_label: int,
            save_name: str = "delta_spatial.png"
    ):
        """可视化 delta 的空间分布

        Args:
            image: 原始图像 (1, 28, 28)
            deltas: 各层的 delta 值
            prediction: 预测标签
            true_label: 真实标签
            save_name: 保存文件名
        """
        num_layers = len(deltas)
        fig, axes = plt.subplots(2, num_layers + 1, figsize=(4 * (num_layers + 1), 8))

        # 计算 patch 网格大小
        n_patches = deltas['layer_0'].shape[0]
        grid_size = int(np.sqrt(n_patches))

        # 第一行第一列：显示原图
        ax = axes[0, 0]
        # ax.imshow(image.squeeze(), cmap='gray')
        display_img = self.process_image_for_plot(image)
        ax.imshow(display_img)

        ax.set_title(f'Original\nTrue: {true_label}, Pred: {prediction}',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

        # 第二行第一列：空白
        axes[1, 0].axis('off')

        # 遍历每一层
        for layer_idx in range(num_layers):
            delta = deltas[f'layer_{layer_idx}']  # (n_patches, d_inner)

            # 计算每个 patch 的平均 delta（跨通道平均）
            delta_mean = delta.mean(dim=1).numpy()  # (n_patches,)
            delta_std = delta.std(dim=1).numpy()  # (n_patches,)

            # 重塑为 2D 网格
            delta_mean_2d = delta_mean.reshape(grid_size, grid_size)
            delta_std_2d = delta_std.reshape(grid_size, grid_size)

            # 第一行：平均 delta 值
            ax = axes[0, layer_idx + 1]
            im = ax.imshow(delta_mean_2d, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Layer {layer_idx}\nDelta Mean', fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 第二行：delta 标准差
            ax = axes[1, layer_idx + 1]
            im = ax.imshow(delta_std_2d, cmap='plasma', interpolation='nearest')
            ax.set_title(f'Layer {layer_idx}\nDelta Std', fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle('Delta Spatial Distribution', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Delta spatial distribution saved: {save_path}")

    def plot_delta_statistics(
            self,
            deltas_list: List[Dict[str, torch.Tensor]],
            labels: List[int],
            save_name: str = "delta_statistics.png"
    ):
        """统计分析多个样本的 delta 值

        Args:
            deltas_list: 多个样本的 delta 字典列表
            labels: 对应的标签
            save_name: 保存文件名
        """
        num_layers = len(deltas_list[0])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Delta 值分布（各层的直方图）
        ax = axes[0, 0]
        for layer_idx in range(num_layers):
            all_deltas = []
            for deltas in deltas_list:
                delta = deltas[f'layer_{layer_idx}'].numpy().flatten()
                all_deltas.extend(delta)

            ax.hist(all_deltas, bins=50, alpha=0.6, label=f'Layer {layer_idx}', density=True)

        ax.set_xlabel('Delta Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Delta Value Distribution (Layer Comparison)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2. Delta 均值随层变化
        ax = axes[0, 1]
        layer_means = []
        layer_stds = []

        for layer_idx in range(num_layers):
            all_deltas = []
            for deltas in deltas_list:
                delta = deltas[f'layer_{layer_idx}'].numpy().flatten()
                all_deltas.extend(delta)

            layer_means.append(np.mean(all_deltas))
            layer_stds.append(np.std(all_deltas))

        x = np.arange(num_layers)
        ax.errorbar(x, layer_means, yerr=layer_stds, marker='o', linewidth=2,
                    markersize=8, capsize=5, capthick=2)
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Delta Mean ± Std', fontsize=12, fontweight='bold')
        ax.set_title('Delta Statistics vs Layer Depth', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)

        # 3. 不同类别的 Delta 模式
        ax = axes[1, 0]
        class_deltas = {i: [] for i in range(10)}

        for deltas, label in zip(deltas_list, labels):
            # 计算所有层的平均 delta
            all_layer_deltas = []
            for layer_idx in range(num_layers):
                delta = deltas[f'layer_{layer_idx}'].mean().item()
                all_layer_deltas.append(delta)
            class_deltas[label].append(np.mean(all_layer_deltas))

        class_means = [np.mean(class_deltas[i]) if len(class_deltas[i]) > 0 else 0
                       for i in range(10)]

        bars = ax.bar(range(10), class_means, color='steelblue', edgecolor='black')
        ax.set_xlabel('Digit Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Delta Value', fontsize=12, fontweight='bold')
        ax.set_title('Delta Response by Class', fontsize=13, fontweight='bold')
        ax.set_xticks(range(10))
        ax.grid(True, axis='y', alpha=0.3)

        # 4. Delta 变化幅度（层间差异）
        ax = axes[1, 1]
        layer_changes = []

        for i in range(num_layers - 1):
            changes = []
            for deltas in deltas_list:
                delta_curr = deltas[f'layer_{i}'].mean().item()
                delta_next = deltas[f'layer_{i + 1}'].mean().item()
                changes.append(abs(delta_next - delta_curr))
            layer_changes.append(changes)

        bp = ax.boxplot(layer_changes, labels=[f'{i}→{i + 1}' for i in range(num_layers - 1)])
        ax.set_xlabel('Layer Transition', fontsize=12, fontweight='bold')
        ax.set_ylabel('|Delta Change|', fontsize=12, fontweight='bold')
        ax.set_title('Inter-layer Delta Change', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        plt.suptitle('Delta Statistical Analysis', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Delta statistics saved: {save_path}")

    def plot_delta_heatmap(
            self,
            image: np.ndarray,
            deltas: Dict[str, torch.Tensor],
            prediction: int,
            true_label: int,
            save_name: str = "delta_heatmap.png"
    ):
        """生成 delta 热力图叠加到原图

        Args:
            image: 原始图像
            deltas: delta 值
            prediction: 预测
            true_label: 真实标签
            save_name: 保存文件名
        """
        num_layers = len(deltas)
        fig, axes = plt.subplots(2, num_layers // 2, figsize=(16, 8))
        axes = axes.flatten()

        n_patches = deltas['layer_0'].shape[0]
        grid_size = int(np.sqrt(n_patches))

        for layer_idx in range(num_layers):
            ax = axes[layer_idx]

            # 计算 delta 均值并重塑
            delta = deltas[f'layer_{layer_idx}'].mean(dim=1).numpy()
            delta_2d = delta.reshape(grid_size, grid_size)

            # 上采样到原图大小
            delta_upsampled = zoom(delta_2d, 28 / grid_size, order=1)

            # 显示原图
            ax.imshow(image.squeeze(), cmap='gray', alpha=0.5)

            # 叠加 delta 热力图
            im = ax.imshow(delta_upsampled, cmap='hot', alpha=0.5)
            ax.set_title(f'Layer {layer_idx} Delta Heatmap', fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'Delta Heatmap Overlay (True: {true_label}, Pred: {prediction})',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Delta heatmap saved: {save_path}")

    def plot_delta_heatmap_extended(
            self,
            image: np.ndarray,
            deltas: Dict[str, torch.Tensor],
            prediction: int,
            true_label: int,
            save_name: str = "delta_heatmap_extended.png"
    ):
        """扩展版：同时显示均值和Top-3通道"""
        num_layers = len(deltas)
        n_patches = deltas['layer_0'].shape[0]
        grid_size = int(np.sqrt(n_patches))

        # 每层5列：原图 + 均值 + Top-3通道
        fig, axes = plt.subplots(num_layers, 5, figsize=(20, 4 * num_layers))

        if num_layers == 1:
            axes = axes.reshape(1, -1)

        for layer_idx in range(num_layers):
            delta_raw = deltas[f'layer_{layer_idx}']

            # 原图
            # axes[layer_idx, 0].imshow(image.squeeze(), cmap='gray')
            display_img = self.process_image_for_plot(image)
            axes[layer_idx, 0].imshow(display_img)

            axes[layer_idx, 0].set_title(f'Layer {layer_idx}\nOriginal', fontsize=10)
            axes[layer_idx, 0].axis('off')

            # 均值图（所有通道平均）
            delta_mean = delta_raw.mean(dim=1).numpy()
            delta_2d = delta_mean.reshape(grid_size, grid_size)
            delta_upsampled = zoom(delta_2d, 28 / grid_size, order=1)

            ax = axes[layer_idx, 1]
            # ax.imshow(image.squeeze(), cmap='gray', alpha=0.5)
            ax.imshow(display_img, alpha=0.5)

            im = ax.imshow(delta_upsampled, cmap='hot', alpha=0.5)
            ax.set_title('All Channels\nMean', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Top-3 通道
            channel_variance = delta_raw.var(dim=0)
            top_channels = torch.topk(channel_variance, k=3).indices

            for i, channel_idx in enumerate(top_channels):
                single_channel_delta = delta_raw[:, channel_idx].numpy()
                delta_2d = single_channel_delta.reshape(grid_size, grid_size)
                delta_upsampled = zoom(delta_2d, 28 / grid_size, order=1)

                ax = axes[layer_idx, i + 2]
                # ax.imshow(image.squeeze(), cmap='gray', alpha=0.5)
                ax.imshow(display_img, alpha=0.5)
                
                im = ax.imshow(delta_upsampled, cmap='hot', alpha=0.5)

                ch_var = channel_variance[channel_idx].item()
                ax.set_title(f'Ch {channel_idx.item()}\nVar={ch_var:.4f}',
                             fontsize=9, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'Delta Heatmap: Mean vs Top-3 High-Variance Channels\n'
                     f'(True: {true_label}, Pred: {prediction})',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Delta extended heatmap saved: {save_path}")

    def plot_delta_channel_analysis(
            self,
            deltas: Dict[str, torch.Tensor],
            save_name: str = "delta_channel_analysis.png"
    ):
        """分析 delta 在不同通道的分布

        Args:
            deltas: delta 值
            save_name: 保存文件名
        """
        num_layers = len(deltas)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))

        if num_layers == 1:
            axes = [axes]

        for layer_idx in range(num_layers):
            ax = axes[layer_idx]
            delta = deltas[f'layer_{layer_idx}'].numpy()  # (n_patches, d_inner)

            # 计算每个通道的统计量
            channel_means = delta.mean(axis=0)  # (d_inner,)
            channel_stds = delta.std(axis=0)  # (d_inner,)

            # 绘制每个通道的分布
            x = np.arange(len(channel_means))
            ax.fill_between(x, channel_means - channel_stds, channel_means + channel_stds,
                            alpha=0.3, color='blue')
            ax.plot(x, channel_means, linewidth=2, color='darkblue')

            ax.set_xlabel('Channel Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Delta Value', fontsize=12, fontweight='bold')
            ax.set_title(f'Layer {layer_idx}\nChannel-wise Delta Distribution',
                         fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Delta Channel-level Analysis', fontsize=15, fontweight='bold', y=1.00)
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Delta channel analysis saved: {save_path}")

    def process_image_for_plot(self, image):
        """
        将 (C, H, W) 的 image 处理为 imshow 可用的 (H, W, C)
        并且进行反归一化以便显示正常颜色
        """
        # 1. 移除 batch 维度 (如果存在) -> (C, H, W)
        if len(image.shape) == 4:
            image = image.squeeze(0)
        
        # 2. Transpose: (C, H, W) -> (H, W, C)
        if isinstance(image, torch.Tensor):
            image = image.cpu().permute(1, 2, 0).numpy()
        elif isinstance(image, np.ndarray):
            # 如果已经是 numpy，检查 shape
            if image.shape[0] == 3: # (3, 32, 32)
                image = np.transpose(image, (1, 2, 0))
        
        # 3. 反归一化 (针对 CIFAR-10 的 mean/std)
        # mean = (0.4914, 0.4822, 0.4465)
        # std = (0.2023, 0.1994, 0.2010)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        
        image = std * image + mean
        image = np.clip(image, 0, 1) # 限制在 0-1 之间
        
        return image



def load_model(model_path: str, device: torch.device) -> VisionMamba:
    """加载训练好的模型"""
    print(f"📂 Loading model: {model_path}")
    model = create_vision_mamba_mnist(d_model=128, patch_size=1)
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded successfully")

    return model

def get_test_loader(batch_size: int = 1, data_dir: str = "./data") -> DataLoader:
    """创建测试数据加载器 (已修正为 CIFAR-10 以匹配训练脚本)"""
    
    # 1. 使用 train.py 中定义的 CIFAR-10 均值和标准差
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    # 2. 修改为 CIFAR10 数据集，而不是 FashionMNIST
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return test_loader


# def get_test_loader(batch_size: int = 1, data_dir: str = "./data") -> DataLoader:
#     """创建测试数据加载器"""
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     test_dataset = datasets.FashionMNIST(
#         root=data_dir,
#         train=False,
#         download=True,
#         transform=transform
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=True
#     )

#     return test_loader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Mamba Delta Visualization')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='./delta_viz',
                        help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")

    # 配置字体
    print("🔧 Configuring fonts...")
    configure_fonts()

    # 加载模型
    model = load_model(args.model_path, device)

    # 创建可视化器
    visualizer = DeltaVisualizer(model, device, args.save_dir)

    # 加载测试数据
    print("\n📦 Loading test data...")
    test_loader = get_test_loader(batch_size=1, data_dir=args.data_dir)

    print(f"\n{'=' * 60}")
    print("🎨 Starting Delta Visualization")
    print(f"{'=' * 60}\n")

    # 收集样本的 delta 值
    deltas_list = []
    labels_list = []
    images_list = []
    predictions_list = []

    for idx, (image, label) in enumerate(tqdm(test_loader, desc="Extracting Delta", total=args.num_samples)):
        if idx >= args.num_samples:
            break

        image = image.to(device)

        # 提取 delta
        deltas = visualizer.extract_delta_values(image)

        # 获取预测
        with torch.no_grad():
            logits = model(image)
            prediction = logits.argmax(dim=1).item()

        deltas_list.append(deltas)
        labels_list.append(label.item())
        images_list.append(image.cpu().numpy())
        predictions_list.append(prediction)

    # 1. 可视化单个样本的空间分布
    print("\n1/5 Generating Delta spatial distribution...")
    for i in range(min(4, len(images_list))):
        visualizer.plot_delta_spatial_distribution(
            images_list[i],
            deltas_list[i],
            predictions_list[i],
            labels_list[i],
            save_name=f"delta_spatial_sample_{i}.png"
        )

    # 2. Delta 统计分析
    print("2/5 Generating Delta statistics...")
    visualizer.plot_delta_statistics(deltas_list, labels_list)

    # 3. Delta 热力图
    print("3/5 Generating Delta heatmaps...")
    for i in range(min(4, len(images_list))):
        visualizer.plot_delta_heatmap_extended(
            images_list[i],
            deltas_list[i],
            predictions_list[i],
            labels_list[i],
            save_name=f"delta_heatmap_extended_sample_{i}.png"
        )

    # 4. Delta 通道分析
    print("4/5 Generating Delta channel analysis...")
    for i in range(min(2, len(deltas_list))):
        visualizer.plot_delta_channel_analysis(
            deltas_list[i],
            save_name=f"delta_channel_sample_{i}.png"
        )

    print(f"\n{'=' * 60}")
    print(f"✨ Delta visualization complete!")
    print(f"📁 Results saved in: {args.save_dir}")
    print(f"{'=' * 60}\n")

    # 打印一些统计信息
    print("📊 Delta Statistics Summary:")
    for layer_idx in range(len(deltas_list[0])):
        all_deltas = []
        for deltas in deltas_list:
            delta = deltas[f'layer_{layer_idx}'].numpy().flatten()
            all_deltas.extend(delta)

        print(f"  Layer {layer_idx}:")
        print(f"    Mean: {np.mean(all_deltas):.6f}")
        print(f"    Std:  {np.std(all_deltas):.6f}")
        print(f"    Min:  {np.min(all_deltas):.6f}")
        print(f"    Max:  {np.max(all_deltas):.6f}")


if __name__ == "__main__":
    main()
