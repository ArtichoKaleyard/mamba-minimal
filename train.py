"""Vision Mamba MNIST训练脚本

完整的训练流程，包括：
    - 数据加载与预处理
    - 模型训练与验证
    - 学习率调度
    - 模型保存与加载
    - 训练过程可视化
    - 日志记录

使用方法:
    python train.py --epochs 5 --batch_size 256 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import time
from tqdm import tqdm
from typing import Tuple, Dict
import json
from pathlib import Path

from vision_mamba import create_vision_mamba_mnist, VisionMamba


class Trainer:
    """训练器类：封装完整的训练流程"""

    def __init__(
            self,
            model: VisionMamba,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            scheduler: optim.lr_scheduler._LRScheduler,
            device: torch.device,
            save_dir: str = "./checkpoints",
            log_dir: str = "./logs"
    ):
        """初始化训练器

        Args:
            model: Vision Mamba模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 训练设备 (cuda/cpu)
            save_dir: 模型保存目录
            log_dir: 日志保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # 创建保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir)

        # 训练统计
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch

        Args:
            epoch: 当前epoch编号

        Returns:
            avg_loss: 平均训练损失
            avg_acc: 平均训练准确率
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            # 数据移到设备
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

            # 记录到TensorBoard（每100个batch记录一次）
            if batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        avg_loss = running_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """验证模型性能

        Args:
            epoch: 当前epoch编号

        Returns:
            avg_loss: 平均验证损失
            avg_acc: 平均验证准确率
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = running_loss / len(self.val_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点

        Args:
            epoch: 当前epoch编号
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }

        # 保存最新模型
        latest_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        print(f"📂 加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']

        return checkpoint['epoch']

    def train(self, num_epochs: int, resume: str = None):
        """完整训练流程

        Args:
            num_epochs: 训练轮数
            resume: 恢复训练的检查点路径（可选）
        """
        start_epoch = 0

        # 如果指定了恢复训练
        if resume:
            start_epoch = self.load_checkpoint(resume) + 1
            print(f"从epoch {start_epoch}继续训练")

        print(f"\n{'=' * 60}")
        print(f"开始训练 Vision Mamba on MNIST")
        print(f"设备: {self.device}")
        print(f"总epochs: {num_epochs}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"{'=' * 60}\n")

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # 验证
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 计算epoch用时
            epoch_time = time.time() - epoch_start_time

            # 记录到TensorBoard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # 打印epoch总结
            print(f"\nEpoch {epoch} 总结:")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")
            print(f"  用时: {epoch_time:.2f}s")

            # 检查是否为最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"  🎉 新的最佳验证准确率: {val_acc:.2f}%")

            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            print(f"{'-' * 60}\n")

        print(f"\n{'=' * 60}")
        print(f"训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"{'=' * 60}\n")

        # 保存训练历史
        self.save_training_history()

        self.writer.close()

    def save_training_history(self):
        """保存训练历史数据"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }

        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"📊 训练历史已保存: {history_path}")


def get_data_loaders(
        batch_size: int = 128,
        num_workers: int = 4,
        data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """创建MNIST数据加载器

    Args:
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        data_dir: 数据集保存目录

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 1. 均值和标准差变了 (RGB 3通道)
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    # 2. 数据增强与标准化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 经典的 CIFAR 增强
        transforms.RandomHorizontalFlip(),    # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    # 3. 加载数据集 (改为 CIFAR10)
    train_dataset = datasets.CIFAR10( # 原来是 FashionMNIST
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR10( # 原来是 FashionMNIST
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )


    # # 数据增强与标准化
    # train_transform = transforms.Compose([
    #     transforms.RandomRotation(10),  # 随机旋转±10度
    #     transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    # ])

    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    # # 加载数据集
    # train_dataset = datasets.FashionMNIST(
    #     root=data_dir,
    #     train=True,
    #     download=True,
    #     transform=train_transform
    # )

    # val_dataset = datasets.FashionMNIST(
    #     root=data_dir,
    #     train=False,
    #     download=True,
    #     transform=val_transform
    # )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Vision Mamba on MNIST')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=128,
                        help='隐藏层维度 (默认: 128)')
    parser.add_argument('--n_layer', type=int, default=4,
                        help='Mamba层数 (默认: 4)')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='图像块大小 (默认: 4)')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='Dropout率 (默认: 0.1)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数 (默认: 20)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小 (默认: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率 (默认: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减 (默认: 1e-4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载工作进程数 (默认: 4)')

    # 路径参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集目录 (默认: ./data)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录 (默认: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志保存目录 (默认: ./logs)')

    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径 (可选)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (默认: cuda)')

    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """主函数：完整的训练流程"""
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU训练")

    # 创建数据加载器
    print("📦 加载MNIST数据集...")
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    # 创建模型
    print("🔨 创建Vision Mamba模型...")
    model = create_vision_mamba_mnist(
        d_model=args.d_model,
        n_layer=args.n_layer,
        patch_size=args.patch_size,
        drop_rate=args.drop_rate
    )

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 定义学习率调度器（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )

    # 开始训练
    trainer.train(num_epochs=args.epochs, resume=args.resume)


if __name__ == "__main__":
    main()