import torch
import numpy as np
import os
import json

print("=== 步骤1：提取适配器权重 ===")

# 加载适配器模型
adapter_path = "/root/千问白盒化实验/models/adapter/adapter_model.pt"

try:
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
except:
    checkpoint = torch.load(adapter_path, map_location="cpu")

state_dict = checkpoint['model_state_dict']

print("适配器权重键名:")
for key in state_dict.keys():
    shape = state_dict[key].shape
    print(f"  {key}: {shape}")

# 我们的适配器是两层：896→256→128
# 提取第一层权重和偏置
fc1_weight = state_dict["network.0.weight"].cpu().numpy().astype(np.float32)  # (256, 896)
fc1_bias = state_dict["network.0.bias"].cpu().numpy().astype(np.float32)      # (256,)

# 提取第二层权重和偏置
fc2_weight = state_dict["network.3.weight"].cpu().numpy().astype(np.float32)  # (128, 256)
fc2_bias = state_dict["network.3.bias"].cpu().numpy().astype(np.float32)      # (128,)

# LayerNorm参数
ln_weight = state_dict["network.4.weight"].cpu().numpy().astype(np.float32)   # (128,)
ln_bias = state_dict["network.4.bias"].cpu().numpy().astype(np.float32)       # (128,)

print(f"\n维度信息:")
print(f"  第一层: {fc1_weight.shape[1]} → {fc1_weight.shape[0]}")
print(f"  第二层: {fc2_weight.shape[1]} → {fc2_weight.shape[0]}")
print(f"  LayerNorm: {ln_weight.shape[0]}")

# 创建输出目录
output_dir = "/root/千问白盒化实验/c_adapter"
os.makedirs(output_dir, exist_ok=True)

# 保存为二进制文件（C语言使用）
print(f"\n保存二进制文件到: {output_dir}")

# 第一层
with open(f"{output_dir}/fc1_weight.bin", "wb") as f:
    f.write(fc1_weight.tobytes())
print(f"  fc1_weight.bin: {fc1_weight.shape} ({fc1_weight.size}个值)")

with open(f"{output_dir}/fc1_bias.bin", "wb") as f:
    f.write(fc1_bias.tobytes())
print(f"  fc1_bias.bin: {fc1_bias.shape} ({fc1_bias.size}个值)")

# 第二层
with open(f"{output_dir}/fc2_weight.bin", "wb") as f:
    f.write(fc2_weight.tobytes())
print(f"  fc2_weight.bin: {fc2_weight.shape} ({fc2_weight.size}个值)")

with open(f"{output_dir}/fc2_bias.bin", "wb") as f:
    f.write(fc2_bias.tobytes())
print(f"  fc2_bias.bin: {fc2_bias.shape} ({fc2_bias.size}个值)")

# LayerNorm
with open(f"{output_dir}/ln_weight.bin", "wb") as f:
    f.write(ln_weight.tobytes())
print(f"  ln_weight.bin: {ln_weight.shape} ({ln_weight.size}个值)")

with open(f"{output_dir}/ln_bias.bin", "wb") as f:
    f.write(ln_bias.tobytes())
print(f"  ln_bias.bin: {ln_bias.shape} ({ln_bias.size}个值)")

# 保存维度信息
dims = {
    "input_dim": int(fc1_weight.shape[1]),  # 896
    "hidden_dim": int(fc1_weight.shape[0]), # 256
    "output_dim": int(fc2_weight.shape[0]), # 128
    "total_params": int(fc1_weight.size + fc1_bias.size + fc2_weight.size + fc2_bias.size + ln_weight.size + ln_bias.size)
}

with open(f"{output_dir}/adapter_dims.json", "w") as f:
    json.dump(dims, f, indent=2)

# 也保存为文本文件供C代码读取
with open(f"{output_dir}/adapter_dims.txt", "w") as f:
    f.write(f"{dims['input_dim']} {dims['hidden_dim']} {dims['output_dim']}")

print(f"\n✅ 权重提取完成!")
print(f"   输入维度: {dims['input_dim']}")
print(f"   隐藏维度: {dims['hidden_dim']}")
print(f"   输出维度: {dims['output_dim']}")
print(f"   总参数: {dims['total_params']:,}")

# 验证文件大小
print(f"\n文件大小验证:")
for filename in ["fc1_weight.bin", "fc1_bias.bin", "fc2_weight.bin", "fc2_bias.bin", "ln_weight.bin", "ln_bias.bin"]:
    filepath = f"{output_dir}/{filename}"
    size_bytes = os.path.getsize(filepath)
    expected_size = {
        "fc1_weight.bin": 256 * 896 * 4,  # 256×896×4 bytes
        "fc1_bias.bin": 256 * 4,
        "fc2_weight.bin": 128 * 256 * 4,
        "fc2_bias.bin": 128 * 4,
        "ln_weight.bin": 128 * 4,
        "ln_bias.bin": 128 * 4
    }[filename]
    
    status = "✅" if size_bytes == expected_size else "❌"
    print(f"  {status} {filename}: {size_bytes:,} bytes (期望: {expected_size:,})")