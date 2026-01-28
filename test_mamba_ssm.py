import torch

# 1. 检查 CUDA 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != 'cuda':
    print("❌ Error: You need a GPU to test Mamba kernels!")
    exit()

print(f"Device: {torch.cuda.get_device_name(0)}")

# 2. 尝试导入官方库
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("✅ Success: 'mamba_ssm' library imported successfully!")
except ImportError as e:
    print(f"❌ Error: Could not import mamba_ssm. Reason: {e}")
    exit()

# 3. 构造虚拟数据测试运行 (基于 1.2.0 版本 API)
# 形状定义: (Batch, Seq_len, Dim)
B, L, D = 2, 64, 16 
N = 16 # State dim

print(f"\nRunning dummy test with shape (B={B}, L={L}, D={D}, N={N})...")

try:
    # 构造随机输入
    u = torch.randn(B, L, D, device=device).contiguous()
    delta = torch.randn(B, L, D, device=device).contiguous()
    A = -torch.rand(D, N, device=device).contiguous()
    B_mat = torch.randn(B, L, N, device=device).contiguous()
    C = torch.randn(B, L, N, device=device).contiguous()
    D_vec = torch.randn(D, device=device).contiguous()

    # 运行官方内核
    # 这是一个非常底层的调用，如果能跑通，说明 CUDA Kernel 编译没问题
    y = selective_scan_fn(
        u, delta, A, B_mat, C, D_vec,
        z=None,
        delta_bias=None,
        delta_softplus=True
    )
    
    print(f"Output shape: {y.shape}")
    print("✨✨✨ CONGRATULATIONS! Mamba CUDA Kernel is working perfectly! ✨✨✨")

except Exception as e:
    print(f"❌ Runtime Error running the kernel: {e}")
    print("Possible reasons: Version mismatch or CUDA driver issues.")
