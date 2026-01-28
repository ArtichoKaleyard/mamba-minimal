"""
基于LSTM的残差块实现，用于与Mamba模型对比
包含卷积层和门控信息提取功能，便于可视化LSTM的三大门控机制
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange
from typing import Dict


# ============ 全局开关 ============
ENABLE_PYTORCH_LSTM = True  # True: 使用PyTorch官方LSTM, False: 使用自定义LSTM
# ==================================


@dataclass
class LSTMModelArgs:
    """LSTM模型参数配置"""
    d_model: int          # 模型隐藏层维度
    n_layer: int          # 层数
    vocab_size: int       # 词表大小
    expand: int = 2       # 扩展因子，与Mamba保持一致
    d_conv: int = 4       # 卷积核大小，与Mamba保持一致
    pad_vocab_size_multiple: int = 8  # 词表大小填充倍数
    conv_bias: bool = True   # 卷积层是否使用偏置
    bias: bool = False       # 线性层是否使用偏置
    dropout: float = 0.0     # dropout比例

    def __post_init__(self):
        """初始化后处理，计算内部维度"""
        self.d_inner = int(self.expand * self.d_model)

        # 确保词表大小是pad_vocab_size_multiple的倍数
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class LSTMModel(nn.Module):
    """完整的LSTM语言模型"""
    def __init__(self, args: LSTMModelArgs):
        super().__init__()
        self.args = args

        # 词嵌入层
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)

        # 堆叠的残差LSTM块
        self.layers = nn.ModuleList([
            ResidualSimpleLSTMBlock(args) for _ in range(args.n_layer)
        ])

        # 最终的RMS归一化层
        self.norm_f = RMSNorm(args.d_model)

        # 输出投影层（语言模型头）
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

        # 权重绑定：输出层与嵌入层共享权重
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        """
        前向传播

        Args:
            input_ids: 输入token ID，shape (batch_size, seq_len)

        Returns:
            logits: 输出logits，shape (batch_size, seq_len, vocab_size)
        """
        # 词嵌入
        x = self.embedding(input_ids)  # (b, l, d_model)

        # 通过所有LSTM残差块
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        x = self.norm_f(x)

        # 输出投影
        logits = self.lm_head(x)  # (b, l, vocab_size)

        return logits


class ResidualSimpleLSTMBlock(nn.Module):
    """
    带残差连接的简化LSTM块
    结构: LayerNorm -> SimpleLSTMBlock -> 残差连接
    """
    def __init__(self, args: LSTMModelArgs):
        super().__init__()
        self.args = args

        # LSTM核心块
        self.lstm_block = SimpleLSTMBlock(args)

        # RMS归一化层
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        """
        前向传播，带残差连接

        Args:
            x: 输入张量，shape (batch_size, seq_len, d_model)

        Returns:
            output: 输出张量，shape (batch_size, seq_len, d_model)
        """
        # 先归一化，再通过LSTM块，最后加残差
        output = self.lstm_block(self.norm(x)) + x
        return output


class SimpleLSTMBlock(nn.Module):
    """
    简化的LSTM块，结构与Mamba对齐以便公平对比
    包含：投影 -> 卷积 -> LSTM -> 门控 -> 输出投影
    特点：保存门控信息用于可视化
    支持切换PyTorch官方LSTM和自定义LSTM
    """
    def __init__(self, args: LSTMModelArgs):
        super().__init__()
        self.args = args

        # 输入投影：将d_model投影到2*d_inner (x分支和残差分支)
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        # 1D卷积层（与Mamba保持一致，使用分组卷积）
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            kernel_size=args.d_conv,
            groups=args.d_inner,  # 深度可分离卷积
            padding=args.d_conv - 1,  # 因果填充
            bias=args.conv_bias
        )

        # LSTM核心层（根据开关选择）
        if ENABLE_PYTORCH_LSTM:
            # 使用PyTorch官方LSTM
            self.lstm = nn.LSTM(
                input_size=args.d_inner,
                hidden_size=args.d_inner,
                num_layers=1,
                batch_first=True,
                bias=True
            )
            print("✓ 使用PyTorch官方LSTM")
        else:
            # 使用自定义LSTM单元
            self.lstm_cell = CustomLSTMCell(args.d_inner, args.d_inner)
            print("✓ 使用自定义LSTM（支持门控可视化）")

        # 输出投影：将d_inner投影回d_model
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

        # 用于存储最近一次前向传播的门控信息（用于可视化）
        self.gate_cache = {}

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，shape (batch_size, seq_len, d_model)

        Returns:
            output: 输出张量，shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 输入投影，分为主分支x和残差分支res
        x_and_res = self.in_proj(x)  # (b, l, 2*d_inner)
        x, res = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        # 2. 卷积操作（转换维度以适应Conv1d）
        x = rearrange(x, 'b l d -> b d l')  # (b, d_inner, l)
        x = self.conv1d(x)[:, :, :seq_len]  # 因果截断，保持序列长度
        x = rearrange(x, 'b d l -> b l d')  # (b, l, d_inner)

        # 3. 激活函数
        x = F.silu(x)

        # 4. 通过LSTM处理序列（根据开关选择不同实现）
        if ENABLE_PYTORCH_LSTM:
            y = self.process_pytorch_lstm(x)
        else:
            y, gate_info = self.process_custom_lstm(x)
            # 保存门控信息用于可视化
            self.gate_cache = gate_info

        # 5. 门控融合：使用残差分支进行门控
        y = y * F.silu(res)

        # 6. 输出投影
        output = self.out_proj(y)

        return output

    def process_pytorch_lstm(self, x):
        """
        使用PyTorch官方LSTM处理序列

        Args:
            x: 输入序列，shape (batch_size, seq_len, d_inner)

        Returns:
            outputs: LSTM输出序列，shape (batch_size, seq_len, d_inner)
        """
        # PyTorch的LSTM期望输入格式为 (batch, seq, feature)，batch_first=True
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs

    def process_custom_lstm(self, x):
        """
        使用自定义LSTM逐时间步处理序列，提取门控信息

        Args:
            x: 输入序列，shape (batch_size, seq_len, d_inner)

        Returns:
            outputs: LSTM输出序列，shape (batch_size, seq_len, d_inner)
            gate_info: 门控信息字典，包含input_gate, forget_gate, output_gate
        """
        batch_size, seq_len, d_inner = x.shape
        device = x.device

        # 初始化隐藏状态和细胞状态
        h = torch.zeros(batch_size, d_inner, device=device)
        c = torch.zeros(batch_size, d_inner, device=device)

        # 用于存储每个时间步的输出和门控信息
        outputs = []
        input_gates = []
        forget_gates = []
        output_gates = []
        cell_gates = []

        # 逐时间步处理
        for t in range(seq_len):
            # 获取当前时间步的输入
            x_t = x[:, t, :]  # (b, d_inner)

            # 通过LSTM单元
            h, c, gates = self.lstm_cell(x_t, h, c)

            # 保存输出和门控信息
            outputs.append(h)
            input_gates.append(gates['input_gate'])
            forget_gates.append(gates['forget_gate'])
            output_gates.append(gates['output_gate'])
            cell_gates.append(gates['cell_gate'])

        # 堆叠所有时间步的结果
        outputs = torch.stack(outputs, dim=1)  # (b, l, d_inner)

        # 组织门控信息
        gate_info = {
            'input_gate': torch.stack(input_gates, dim=1),    # (b, l, d_inner)
            'forget_gate': torch.stack(forget_gates, dim=1),  # (b, l, d_inner)
            'output_gate': torch.stack(output_gates, dim=1),  # (b, l, d_inner)
            'cell_gate': torch.stack(cell_gates, dim=1)       # (b, l, d_inner)
        }

        return outputs, gate_info

    def get_gate_info(self) -> Dict[str, torch.Tensor]:
        """
        获取最近一次前向传播的门控信息
        注意：仅在使用自定义LSTM时有效

        Returns:
            gate_cache: 包含input_gate, forget_gate, output_gate的字典
        """
        if ENABLE_PYTORCH_LSTM:
            print("警告：PyTorch官方LSTM不支持门控信息提取")
            return {}
        return self.gate_cache


class CustomLSTMCell(nn.Module):
    """
    自定义LSTM单元，便于提取和可视化门控信息
    标准LSTM公式：
        i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)  # 输入门
        f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)  # 遗忘门
        g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)  # 候选细胞状态
        o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)  # 输出门
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  # 细胞状态更新
        h_t = o_t ⊙ tanh(c_t)  # 隐藏状态输出
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入到隐藏的权重矩阵（合并四个门的权重以提高效率）
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)

        # 隐藏到隐藏的权重矩阵（合并四个门的权重）
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, h_prev, c_prev):
        """
        LSTM单元的前向传播

        Args:
            x: 当前时间步输入，shape (batch_size, input_size)
            h_prev: 上一时间步隐藏状态，shape (batch_size, hidden_size)
            c_prev: 上一时间步细胞状态，shape (batch_size, hidden_size)

        Returns:
            h_new: 新的隐藏状态，shape (batch_size, hidden_size)
            c_new: 新的细胞状态，shape (batch_size, hidden_size)
            gates: 门控信息字典
        """
        # 计算所有门的预激活值
        gates_input = self.weight_ih(x)       # (b, 4*hidden_size)
        gates_hidden = self.weight_hh(h_prev)  # (b, 4*hidden_size)
        gates_combined = gates_input + gates_hidden  # (b, 4*hidden_size)

        # 分割为四个门
        i, f, g, o = gates_combined.chunk(4, dim=1)

        # 应用激活函数
        input_gate = torch.sigmoid(i)    # 输入门
        forget_gate = torch.sigmoid(f)   # 遗忘门
        cell_gate = torch.tanh(g)        # 候选细胞状态（也称为cell gate）
        output_gate = torch.sigmoid(o)   # 输出门

        # 更新细胞状态
        c_new = forget_gate * c_prev + input_gate * cell_gate

        # 计算新的隐藏状态
        h_new = output_gate * torch.tanh(c_new)

        # 收集门控信息用于可视化
        gates_info = {
            'input_gate': input_gate,
            'forget_gate': forget_gate,
            'cell_gate': cell_gate,
            'output_gate': output_gate
        }

        return h_new, c_new, gates_info


class RMSNorm(nn.Module):
    """
    RMS归一化层（Root Mean Square Normalization）
    与LayerNorm类似但更简单高效
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        RMS归一化

        Args:
            x: 输入张量，shape (..., d_model)

        Returns:
            output: 归一化后的张量，shape (..., d_model)
        """
        # 计算RMS并归一化
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = x * rms * self.weight
        return output


# 工具函数：用于可视化门控信息
def visualize_gates(lstm_block: SimpleLSTMBlock,
                    seq_len: int = None,
                    save_path: str = None):
    """
    可视化LSTM门控信息的工具函数

    Args:
        lstm_block: SimpleLSTMBlock实例
        seq_len: 要可视化的序列长度（None表示全部）
        save_path: 保存图像的路径（None表示不保存）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 获取门控信息
    gate_info = lstm_block.get_gate_info()

    if not gate_info:
        print("警告：没有可用的门控信息，请先进行前向传播")
        if ENABLE_PYTORCH_LSTM:
            print("提示：PyTorch官方LSTM不支持门控可视化，请设置 ENABLE_PYTORCH_LSTM = False")
        return

    # 提取数据并转换为numpy（取第一个batch和平均值）
    input_gate = gate_info['input_gate'][0].mean(dim=-1).detach().cpu().numpy()
    forget_gate = gate_info['forget_gate'][0].mean(dim=-1).detach().cpu().numpy()
    output_gate = gate_info['output_gate'][0].mean(dim=-1).detach().cpu().numpy()

    if seq_len is not None:
        input_gate = input_gate[:seq_len]
        forget_gate = forget_gate[:seq_len]
        output_gate = output_gate[:seq_len]

    # 绘制三个门控的激活值
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(input_gate, label='Input Gate', color='blue')
    axes[0].set_ylabel('激活值')
    axes[0].set_title('输入门 (Input Gate)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(forget_gate, label='Forget Gate', color='red')
    axes[1].set_ylabel('激活值')
    axes[1].set_title('遗忘门 (Forget Gate)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(output_gate, label='Output Gate', color='green')
    axes[2].set_xlabel('时间步')
    axes[2].set_ylabel('激活值')
    axes[2].set_title('输出门 (Output Gate)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"门控可视化已保存到: {save_path}")

    plt.show()


# 示例使用代码
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"LSTM模式: {'PyTorch官方LSTM' if ENABLE_PYTORCH_LSTM else '自定义LSTM（支持门控可视化）'}")
    print(f"{'='*60}\n")

    # 创建模型配置
    args = LSTMModelArgs(
        d_model=256,
        n_layer=4,
        vocab_size=1000,
        expand=2,
        d_conv=4
    )

    # 创建模型
    model = LSTMModel(args)

    # 创建示例输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    # 前向传播
    logits = model(input_ids)
    print(f"输出shape: {logits.shape}")  # 应该是 (2, 10, 1000)

    # 提取第一个LSTM块的门控信息
    first_lstm_block = model.layers[0].lstm_block
    gate_info = first_lstm_block.get_gate_info()

    if gate_info:
        print("\n门控信息shapes:")
        for gate_name, gate_values in gate_info.items():
            print(f"{gate_name}: {gate_values.shape}")

        # 可视化门控（需要安装matplotlib）
        try:
            visualize_gates(first_lstm_block, seq_len=seq_len)
        except ImportError:
            print("\n提示：安装matplotlib以启用门控可视化功能")
    else:
        print("\n提示：当前使用PyTorch官方LSTM，不支持门控信息提取")
        print("如需可视化门控，请设置 ENABLE_PYTORCH_LSTM = False")
