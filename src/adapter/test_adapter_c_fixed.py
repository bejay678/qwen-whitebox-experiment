import ctypes
import numpy as np
import torch
import time
import sys
import os

print("=== 步骤3：Python调用C库并验证数值一致性 ===")

# 加载共享库
lib_path = "/root/千问白盒化实验/c_adapter/libadapter.so"
print(f"加载共享库: {lib_path}")

try:
    lib = ctypes.CDLL(lib_path)
    print("✅ 共享库加载成功")
except Exception as e:
    print(f"❌ 共享库加载失败: {e}")
    sys.exit(1)

# 定义函数原型
lib.adapter_init.argtypes = [ctypes.c_char_p]
lib.adapter_init.restype = ctypes.c_int

lib.adapter_forward.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
]
lib.adapter_forward.restype = None

lib.adapter_get_dims.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]
lib.adapter_get_dims.restype = None

lib.adapter_cleanup.argtypes = []
lib.adapter_cleanup.restype = None

# 初始化适配器
weight_dir = "/root/千问白盒化实验/c_adapter"
print(f"初始化适配器，权重目录: {weight_dir}")

ret = lib.adapter_init(weight_dir.encode('utf-8'))
if ret != 0:
    print(f"❌ 适配器初始化失败，错误码: {ret}")
    sys.exit(1)

print("✅ 适配器初始化成功")

# 获取维度
input_dim = ctypes.c_int()
hidden_dim = ctypes.c_int()
output_dim = ctypes.c_int()

lib.adapter_get_dims(ctypes.byref(input_dim), ctypes.byref(hidden_dim), ctypes.byref(output_dim))

input_dim = input_dim.value
hidden_dim = hidden_dim.value
output_dim = output_dim.value

print(f"适配器维度: {input_dim} → {hidden_dim} → {output_dim}")

# 数值一致性验证
print("\n=== 数值一致性验证 ===")

# 加载PyTorch适配器进行对比
print("加载PyTorch适配器...")
adapter_path = "/root/千问白盒化实验/models/adapter/adapter_model.pt"

try:
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
except:
    checkpoint = torch.load(adapter_path, map_location="cpu")

state_dict = checkpoint['model_state_dict']

# 重建PyTorch适配器
from torch import nn

class PyTorchAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln(x)
        return x

pytorch_adapter = PyTorchAdapter()

# 加载权重（需要修复键名）
fixed_state_dict = {}
for key, value in state_dict.items():
    if key == "network.0.weight":
        fixed_state_dict["fc1.weight"] = value
    elif key == "network.0.bias":
        fixed_state_dict["fc1.bias"] = value
    elif key == "network.3.weight":
        fixed_state_dict["fc2.weight"] = value
    elif key == "network.3.bias":
        fixed_state_dict["fc2.bias"] = value
    elif key == "network.4.weight":
        fixed_state_dict["ln.weight"] = value
    elif key == "network.4.bias":
        fixed_state_dict["ln.bias"] = value

pytorch_adapter.load_state_dict(fixed_state_dict)
pytorch_adapter.eval()
print("✅ PyTorch适配器加载成功")

# 生成随机输入
np.random.seed(42)
x_np = np.random.randn(input_dim).astype(np.float32)
x_torch = torch.from_numpy(x_np).unsqueeze(0)  # 添加batch维度

# C适配器前向
output_c = np.zeros(output_dim, dtype=np.float32)
lib.adapter_forward(x_np, output_c)

# PyTorch适配器前向
with torch.no_grad():
    output_torch = pytorch_adapter(x_torch).squeeze(0).numpy()

# 比较结果
print(f"\n输入形状: {x_np.shape}")
print(f"C输出形状: {output_c.shape}")
print(f"PyTorch输出形状: {output_torch.shape}")

# 计算误差
abs_diff = np.abs(output_c - output_torch)
max_abs_diff = np.max(abs_diff)
mean_abs_diff = np.mean(abs_diff)

print(f"\n数值一致性结果:")
print(f"  最大绝对误差: {max_abs_diff:.6e}")
print(f"  平均绝对误差: {mean_abs_diff:.6e}")

# 验证标准
if max_abs_diff < 1e-4:
    print("✅ 数值验证通过！误差在可接受范围内")
else:
    print("⚠️  数值误差较大，需要检查")

# 显示前10个值的对比
print(f"\n前10个输出值对比:")
print(f"{'索引':<6} {'C输出':<12} {'PyTorch输出':<12} {'绝对误差':<12}")
print("-" * 50)
for i in range(min(10, output_dim)):
    print(f"{i:<6} {output_c[i]:<12.6f} {output_torch[i]:<12.6f} {abs_diff[i]:<12.6e}")

# 性能测试
print("\n=== 性能测试 ===")

# PyTorch性能（CPU）
print("PyTorch CPU性能测试...")
x_torch_cpu = torch.from_numpy(x_np).float()

def pytorch_forward():
    with torch.no_grad():
        return pytorch_adapter(x_torch_cpu.unsqueeze(0))

# 预热
for _ in range(100):
    pytorch_forward()

# 正式测试
num_iterations = 10000
start = time.perf_counter()
for _ in range(num_iterations):
    pytorch_forward()
torch_time = (time.perf_counter() - start) / num_iterations * 1000  # ms per query

print(f"PyTorch CPU平均耗时: {torch_time:.4f} ms/查询")

# C性能
print("\nC适配器性能测试...")
x_cpu = x_np.copy()
output_cpu = np.zeros(output_dim, dtype=np.float32)

def c_forward():
    lib.adapter_forward(x_cpu, output_cpu)

# 预热
for _ in range(100):
    c_forward()

# 正式测试
start = time.perf_counter()
for _ in range(num_iterations):
    c_forward()
c_time = (time.perf_counter() - start) / num_iterations * 1000  # ms per query

print(f"C适配器平均耗时: {c_time:.4f} ms/查询")

# 计算加速比
speedup = torch_time / c_time if c_time > 0 else 0
print(f"\n性能对比:")
print(f"  PyTorch CPU: {torch_time:.4f} ms/查询")
print(f"  C适配器: {c_time:.4f} ms/查询")
print(f"  加速比: {speedup:.2f}倍")

# 清理
print("\n清理资源...")
lib.adapter_cleanup()

print("\n=== C适配器固化完成 ===")
print(f"✅ 权重提取完成")
print(f"✅ C代码编译完成")
print(f"✅ 数值一致性验证完成 (最大误差: {max_abs_diff:.2e})")
print(f"✅ 性能测试完成 (加速比: {speedup:.2f}倍)")

# 保存测试结果
import json
result = {
    "test_time": time.strftime('%Y-%m-%d %H:%M:%S'),
    "dimensions": {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim
    },
    "numerical_validation": {
        "max_absolute_error": float(max_abs_diff),
        "mean_absolute_error": float(mean_abs_diff),
        "validation_passed": max_abs_diff < 1e-4
    },
    "performance": {
        "pytorch_cpu_ms": float(torch_time),
        "c_adapter_ms": float(c_time),
        "speedup": float(speedup)
    }
}

result_path = "/root/千问白盒化实验/c_adapter/test_results.json"
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✅ 测试结果已保存: {result_path}")