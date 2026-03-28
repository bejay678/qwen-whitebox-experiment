import torch
import sys
import json
import os

print("=== 千问白盒化实验 - 环境验证 ===")
print(f"验证时间: {sys.version.split()[0]}")

# 1. 基础环境
print("\n=== 基础环境 ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU capability: {torch.cuda.get_device_capability()}")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("❌ CUDA不可用！")

# 2. 检查transformers
print("\n=== Transformers检查 ===")
try:
    import transformers
    print(f"✅ transformers版本: {transformers.__version__}")
except ImportError:
    print("❌ transformers未安装")
    print("安装命令: pip install transformers")

# 3. 检查模型文件
print("\n=== 模型文件检查 ===")
model_path = "/root/千问白盒化实验/models/Qwen2.5-0.5B-Instruct"
if os.path.exists(model_path):
    print(f"✅ 模型目录存在: {model_path}")
    
    # 检查关键文件
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"  ✅ {file}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {file}: 不存在")
    
    # 检查配置文件
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            print(f"✅ 配置文件可解析:")
            print(f"   模型类型: {config.get('model_type')}")
            print(f"   隐藏层大小: {config.get('hidden_size')}")
            print(f"   层数: {config.get('num_hidden_layers')}")
            print(f"   torch_dtype: {config.get('torch_dtype')}")
        except Exception as e:
            print(f"❌ 配置文件解析失败: {e}")
else:
    print(f"❌ 模型目录不存在: {model_path}")

# 4. 检查数据集
print("\n=== 数据集检查 ===")
data_dir = "/root/千问白盒化实验/data"
if os.path.exists(data_dir):
    print(f"✅ 数据目录存在: {data_dir}")
    
    data_files = ["facts_dataset.json", "training_pairs_fixed.json"]
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  ✅ {file}: {size_kb:.1f} KB")
            
            # 尝试解析JSON
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    print(f"     内容: 字典，{len(data)}个键")
                elif isinstance(data, list):
                    print(f"     内容: 列表，{len(data)}个元素")
            except Exception as e:
                print(f"     解析错误: {e}")
        else:
            print(f"  ❌ {file}: 不存在")
else:
    print(f"❌ 数据目录不存在: {data_dir}")

# 5. 检查脚本
print("\n=== 脚本检查 ===")
scripts_dir = "/root/千问白盒化实验/scripts"
if os.path.exists(scripts_dir):
    print(f"✅ 脚本目录存在: {scripts_dir}")
    
    script_files = ["simple_hidden_test.py"]
    for file in script_files:
        file_path = os.path.join(scripts_dir, file)
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  ✅ {file}: {size_kb:.1f} KB")
        else:
            print(f"  ❌ {file}: 不存在")
else:
    print(f"❌ 脚本目录不存在: {scripts_dir}")

# 6. 总结
print("\n=== 环境验证总结 ===")
issues = []

if not torch.cuda.is_available():
    issues.append("CUDA不可用")
if not os.path.exists(model_path):
    issues.append("模型目录不存在")
elif not os.path.exists(os.path.join(model_path, "model.safetensors")):
    issues.append("模型文件不存在")

if issues:
    print(f"❌ 发现 {len(issues)} 个问题:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ 所有检查通过！环境就绪，可以开始实验。")

print("\n=== 下一步建议 ===")
if not issues:
    print("运行测试脚本:")
    print("  cd /root/千问白盒化实验/scripts")
    print("  python simple_hidden_test.py")
else:
    print("请先解决上述问题，然后运行测试脚本。")