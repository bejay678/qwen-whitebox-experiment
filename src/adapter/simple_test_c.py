import ctypes
import numpy as np
import time

print("=== 简化C适配器测试 ===")

# 加载共享库
lib_path = "/root/千问白盒化实验/c_adapter/libadapter.so"
print(f"加载共享库: {lib_path}")

try:
    lib = ctypes.CDLL(lib_path)
    print("✅ 共享库加载成功")
except Exception as e:
    print(f"❌ 共享库加载失败: {e}")
    exit(1)

# 定义函数原型
lib.adapter_init.argtypes = [ctypes.c_char_p]
lib.adapter_init.restype = ctypes.c_int

lib.adapter_forward.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32)
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
print(f"初始化适配器...")

ret = lib.adapter_init(weight_dir.encode('utf-8'))
if ret != 0:
    print(f"❌ 适配器初始化失败")
    exit(1)

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

# 简单测试
print("\n=== 简单测试 ===")

# 生成测试输入
np.random.seed(42)
x = np.random.randn(input_dim).astype(np.float32)
output = np.zeros(output_dim, dtype=np.float32)

print(f"输入向量: shape={x.shape}")
print(f"输出向量: shape={output.shape}")

# 运行适配器
print("运行C适配器...")
lib.adapter_forward(x, output)

print(f"输出前5个值: {output[:5]}")

# 性能测试
print("\n=== 性能测试 ===")

num_iterations = 10000
print(f"运行 {num_iterations} 次前向传播...")

start = time.perf_counter()
for _ in range(num_iterations):
    lib.adapter_forward(x, output)
end = time.perf_counter()

total_time = (end - start) * 1000  # 总毫秒数
avg_time = total_time / num_iterations  # 平均毫秒数

print(f"总耗时: {total_time:.2f} ms")
print(f"平均耗时: {avg_time:.4f} ms/查询")
print(f"每秒查询数: {1000/avg_time:.0f} QPS")

# 清理
print("\n清理资源...")
lib.adapter_cleanup()

print("\n=== 测试完成 ===")
print("✅ C适配器工作正常")