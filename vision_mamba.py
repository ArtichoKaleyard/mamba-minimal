"""Vision Mamba for MNIST Classification

这是一个基于Mamba架构的视觉分类模型，专门用于MNIST手写数字识别。
主要特点：
    1. 将2D图像展平为1D序列
    2. 使用patch embedding将图像块转换为token
    3. 复用Mamba的序列建模能力进行分类

参考论文:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
        https://arxiv.org/abs/2312.00752
"""

import torch
import torch.nn as nn
from typing import Union
from dataclasses import dataclass


ENABLE_POS_EMBED = False  # 是否启用位置编码
ENABLE_SWITCH_LSTM_RES_BLOCK = False  # 是否启用LSTM替代Mamba Block进行对比实验


# 导入原始Mamba模块
if not ENABLE_SWITCH_LSTM_RES_BLOCK:
    from model import ModelArgs, RMSNorm, ResidualBlock
else:
    from model import ModelArgs, RMSNorm
    from lstm import ResidualSimpleLSTMBlock as ResidualBlock  # 用LSTM替代Mamba Block进行对比实验


@dataclass
class VisionMambaArgs:
    """Vision Mamba模型配置参数

    Args:
        img_size: 输入图像尺寸 (MNIST为28x28)
        patch_size: 图像块大小，将图像分割为patch_size x patch_size的小块
        in_channels: 输入通道数 (MNIST为1，RGB图像为3)
        num_classes: 分类类别数 (MNIST为10)
        d_model: 隐藏层维度
        n_layer: Mamba层数
        d_state: 状态空间维度
        expand: 扩展因子
        dt_rank: delta参数的秩
        d_conv: 卷积核大小
        drop_rate: Dropout比率
    """
    img_size: int = 28
    patch_size: int = 4
    in_channels: int = 1
    num_classes: int = 10
    d_model: int = 128
    n_layer: int = 4
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    drop_rate: float = 0.1


class PatchEmbedding(nn.Module):
    """图像块嵌入层

    将输入图像分割成不重叠的patch，并将每个patch投影到d_model维度。
    例如：28x28的图像用4x4的patch分割，得到49个patch (7x7)
    """

    def __init__(self, args: VisionMambaArgs):
        super().__init__()
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.n_patches = (args.img_size // args.patch_size) ** 2
        self.patch_dim = args.in_channels * args.patch_size * args.patch_size

        # 使用卷积层实现patch embedding（等价于线性投影）
        self.proj = nn.Conv2d(
            args.in_channels,
            args.d_model,
            kernel_size=args.patch_size,
            stride=args.patch_size
        )

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, in_channels, img_size, img_size)

        Returns:
            patches: shape (batch_size, n_patches, d_model)
        """
        # 通过卷积获取patch embeddings
        x = self.proj(x)  # (B, d_model, H/patch_size, W/patch_size)

        # 展平空间维度
        x = x.flatten(2)  # (B, d_model, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, d_model)

        return x


class VisionMamba(nn.Module):
    """Vision Mamba模型主体

    架构流程:
        输入图像 -> Patch Embedding -> 位置编码 -> Mamba Blocks -> 全局池化 -> 分类头
    """

    def __init__(self, args: VisionMambaArgs):
        super().__init__()
        self.args = args

        # 1. Patch embedding层：将图像转换为序列
        self.patch_embed = PatchEmbedding(args)

        # 2. 可学习的位置编码
        if ENABLE_POS_EMBED:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.n_patches, args.d_model)
            )
        else:
            self.pos_embed = None

        # 3. Dropout层用于正则化
        self.pos_drop = nn.Dropout(p=args.drop_rate)

        # 4. 构建Mamba backbone
        # 复用原始Mamba的ResidualBlock
        mamba_args = ModelArgs(
            d_model=args.d_model,
            n_layer=args.n_layer,
            vocab_size=args.num_classes,  # 这里vocab_size不会被使用
            d_state=args.d_state,
            expand=args.expand,
            dt_rank=args.dt_rank,
            d_conv=args.d_conv
        )
        self.layers = nn.ModuleList([
            ResidualBlock(mamba_args) for _ in range(args.n_layer)
        ])

        # 5. 最终的归一化层
        self.norm = RMSNorm(args.d_model)

        # 6. 分类头
        # 使用全局平均池化 + 线性层
        self.head = nn.Linear(args.d_model, args.num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # 位置编码使用截断正态分布初始化
        if ENABLE_POS_EMBED:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 分类头使用Xavier初始化
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """前向传播

        Args:
            x: shape (batch_size, in_channels, img_size, img_size)

        Returns:
            logits: shape (batch_size, num_classes)
        """
        # 1. Patch embedding: 图像 -> 序列
        x = self.patch_embed(x)  # (B, n_patches, d_model)

        # 2. 添加位置编码
        if ENABLE_POS_EMBED:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # 3. 通过Mamba层进行序列建模
        for layer in self.layers:
            x = layer(x)

        # 4. 归一化
        x = self.norm(x)

        # 5. 全局平均池化：将序列聚合为单个向量
        x = x.mean(dim=1)  # (B, d_model)

        # 6. 分类
        logits = self.head(x)  # (B, num_classes)

        return logits

    def get_num_params(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vision_mamba_mnist(
        d_model: int = 128,
        n_layer: int = 4,
        patch_size: int = 4,
        drop_rate: float = 0.1
) -> VisionMamba:
    """便捷函数：创建MNIST Vision Mamba模型

    Args:
        d_model: 隐藏层维度，默认128
        n_layer: Mamba层数，默认4
        patch_size: 图像块大小，默认4 (28x28图像分为7x7=49个patch)
        drop_rate: Dropout比率，默认0.1

    Returns:
        model: 配置好的VisionMamba模型

    示例:
        >>> model = create_vision_mamba_mnist(d_model=256, n_layer=6)
        >>> x = torch.randn(32, 1, 28, 28)  # batch_size=32
        >>> logits = model(x)  # (32, 10)
    """
    args = VisionMambaArgs(
        img_size=32,
        patch_size=patch_size,
        in_channels=3,
        num_classes=10,
        d_model=d_model,
        n_layer=n_layer,
        drop_rate=drop_rate
    )

    model = VisionMamba(args)

    print(f"模型创建成功!")
    print(f"参数量: {model.get_num_params():,}")
    print(f"输入尺寸: (B, 1, 28, 28)")
    print(f"输出尺寸: (B, 10)")
    print(f"Patch数量: {model.patch_embed.n_patches}")

    return model


# 示例使用代码
if __name__ == "__main__":
    # 创建模型
    model = create_vision_mamba_mnist(
        d_model=128,
        n_layer=4,
        patch_size=4
    )

    # 测试前向传播
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)

    print("\n" + "=" * 50)
    print("测试前向传播...")
    logits = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")

    # 计算预测类别
    predictions = logits.argmax(dim=1)
    print(f"预测形状: {predictions.shape}")
    print(f"示例预测: {predictions[:5]}")

    print("=" * 50)
    print("Vision Mamba模型测试通过! ✓")