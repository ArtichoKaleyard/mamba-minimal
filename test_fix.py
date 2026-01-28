import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# 1. 设置设备
device = torch.device("cuda")

# 2. 定义维度
B = 2    # Batch size
L = 64   # Sequence length
D = 16   # Model dimension (Inner dim)
N = 16   # State dimension (d_state)

print(f"Running Mamba 1.1.1 Test on {torch.cuda.get_device_name(0)}")
print(f"Shape: B={B}, L={L}, D={D}, N={N}")

# 3. 构造数据 (关键修正：使用 half() 转为 FP16)
# mamba 1.1.1 核心输入必须是 FP16，否则内核可能会报错或误报 Shape 错误
dtype = torch.float16 

# u: (Batch, Dim, Length) - 注意 L 在最后
u = torch.randn(B, D, L, device=device, dtype=dtype)
delta = torch.randn(B, D, L, device=device, dtype=dtype)

# A: (Dim, State) - 这里的参数通常保持 FP32 以获得更好的数值稳定性，但转为 FP16 也可以
A = -torch.rand(D, N, device=device, dtype=torch.float32)

# B, C: (Batch, State, Length)
# 在 1.1.1 版本中，直接使用 (B, N, L) 即可
B_mat = torch.randn(B, N, L, device=device, dtype=dtype)
C_mat = torch.randn(B, N, L, device=device, dtype=dtype)

# D, delta_bias: (Dim,) - 保持 FP32
D_vec = torch.randn(D, device=device, dtype=torch.float32)
delta_bias = torch.randn(D, device=device, dtype=torch.float32)

# 4. 强制连续性 (STRIDE CHECK)
# 这是 Mamba 最常见的报错来源，必须确保内存连续
u = u.contiguous()
delta = delta.contiguous()
A = A.contiguous()
B_mat = B_mat.contiguous()
C_mat = C_mat.contiguous()
D_vec = D_vec.contiguous()
delta_bias = delta_bias.contiguous()

# 5. 打印形状进行最后确认
print("-" * 30)
print(f"u:     {u.shape} {u.dtype}")
print(f"delta: {delta.shape} {delta.dtype}")
print(f"A:     {A.shape} {A.dtype}")
print(f"B:     {B_mat.shape} {B_mat.dtype}")
print(f"C:     {C_mat.shape} {C_mat.dtype}")
print("-" * 30)

try:
    # 6. 调用内核
    y = selective_scan_fn(
        u, 
        delta, 
        A, 
        B_mat, 
        C_mat, 
        D_vec, 
        z=None, 
        delta_bias=delta_bias, 
        delta_softplus=True
    )
    
    print("\n✨✨✨ 成功! Mamba Kernel 运行正常! ✨✨✨")
    print(f"Output shape: {y.shape}")

except Exception as e:
    print(f"\n❌ 依然报错: {e}")
    # 有时候 1.1.1 版本对 A 的类型非常挑剔，如果上面报错，尝试把 A 也转成 half
    if "must have shape" in str(e):
        print("\n[Debug建议] 尝试将 A 和 D 也转为 float16 重试...")
