"""
Optimized Mamba implementation.
Includes automatic fallback to pure PyTorch if CUDA kernel is missing.
"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union

# --- 1. 尝试导入 CUDA 加速内核 ---
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    USE_MAMBA_KERNELS = True
    print("✅ Mamba CUDA 内核已加载，将使用硬件加速模式。")
except ImportError:
    USE_MAMBA_KERNELS = False
    print("⚠️ 未检测到 mamba_ssm，将使用纯 PyTorch 慢速模式。")


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x) 
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    
    def ssm(self, x):
        """
        运行 SSM。
        会自动检测是否使用 CUDA 内核加速。
        """
        (d_in, n) = self.A_log.shape
        
        # 1. 计算参数 A, D (输入无关)
        # 注意: 为了数值稳定性，A 和 D 保持 float32
        A = -torch.exp(self.A_log.float())  # (d_in, n)
        D = self.D.float()

        # 2. 计算参数 delta, B, C (输入相关)
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        
        # 3. 处理 Delta
        delta = self.dt_proj(delta)  # (b, l, d_in)
        # 此时 delta 还是 Linear 的输出 (logits)
        # 官方内核通常期望 delta 通过 Softplus，或者我们传 raw delta 给内核让内核做 Softplus
        # 这里为了兼容性，我们先算出 softplus 后的值
        delta = F.softplus(delta) 

        # --- 分支：选择加速模式或纯 PyTorch 模式 ---
        if USE_MAMBA_KERNELS:
            return self.scanz_cuda(x, delta, A, B, C, D)
        else:
            return self.selective_scan_torch(x, delta, A, B, C, D)


    def scanz_cuda(self, u, delta, A, B, C, D):
        """
        使用 mamba_ssm CUDA 内核
        ⚠️ 核心修正：
        1. 必须转置：Transformer Layout (B, L, D) -> SSM Layout (B, D, L)
        2. 必须 contiguous
        3. 必须 float16/bfloat16 (除了 A 和 D)
        """
        
        # 修正1：转置 (B, L, D) -> (B, D, L)
        u_t = u.transpose(1, 2)
        delta_t = delta.transpose(1, 2)
        B_t = B.transpose(1, 2)
        C_t = C.transpose(1, 2)

        # 修正2 & 3：确保 contiguous 和 数据类型
        # 内核要求输入必须是 half (fp16) 或 bf16 (bf16)，否则会报 Shape 错误
        # A 和 D 保持 fp32
        # dtype_kernel = torch.float16 if u.dtype == torch.float32 else u.dtype
        dtype_kernel = torch.bfloat16
        
        u_t = u_t.to(dtype=dtype_kernel).contiguous()
        delta_t = delta_t.to(dtype=dtype_kernel).contiguous()
        B_t = B_t.to(dtype=dtype_kernel).contiguous()
        C_t = C_t.to(dtype=dtype_kernel).contiguous()
        A_t = A.contiguous()
        D_t = D.contiguous()

        # 调用 CUDA 内核
        # delta_softplus=False 因为我们在传入前已经在 python 里做过 F.softplus 了
        y = selective_scan_fn(
            u_t, delta_t, A_t, B_t, C_t, D_t, 
            z=None,
            delta_bias=None,
            delta_softplus=False 
        )

        # 转置回 (B, L, D)
        return y.transpose(1, 2)


    def selective_scan_torch(self, u, delta, A, B, C, D):
        """纯 PyTorch 实现 (慢速，仅作备份)"""
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # 预计算离散化参数
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # 调用被 JIT 编译的循环
        return _selective_scan_loop(u, deltaA, deltaB_u, C, D)


@torch.jit.script
def _selective_scan_loop(u, deltaA, deltaB_u, C, D):
    b, l, d_in = u.shape
    n = deltaA.shape[-1]
    x = torch.zeros((b, d_in, n), device=u.device, dtype=u.dtype)
    ys = torch.empty((b, l, d_in), device=u.device, dtype=u.dtype)

    for i in range(l):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        # x: (b, d_in, n), C[:, i]: (b, n) -> (b, n, 1)
        y = torch.bmm(x, C[:, i].unsqueeze(-1)).squeeze(-1)
        ys[:, i, :] = y

    return ys + u * D


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
