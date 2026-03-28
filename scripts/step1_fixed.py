import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def analyze_adapter():
    """分析适配器结构和权重"""
    print("=== 步骤1：适配器分析 ===")
    
    adapter_path = "/root/千问白盒化实验/models/adapter/adapter_model.pt"
    
    try:
        checkpoint = torch.load(adapter_path, map_location='cpu', weights_only=True)
        print(f"✅ 适配器加载成功")
    except:
        # 如果weights_only失败，尝试普通加载
        checkpoint = torch.load(adapter_path, map_location='cpu')
        print(f"✅ 适配器加载成功（非weights_only模式）")
    
    # 分析结构
    model_state_dict = checkpoint['model_state_dict']
    
    print(f"\n权重键名:")
    for key in model_state_dict.keys():
        shape = model_state_dict[key].shape
        print(f"  {key}: {shape}")
    
    # 提取权重
    weights = {}
    for key, tensor in model_state_dict.items():
        weights[key] = tensor.numpy()
    
    # 确定维度
    input_dim = 896  # 已知
    output_dim = 128  # 已知
    
    print(f"\n=== 维度确认 ===")
    print(f"输入维度: {input_dim}")
    print(f"输出维度: {output_dim}")
    
    # 保存权重
    weights_dir = Path("/root/千问白盒化实验/models/adapter/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    for key, array in weights.items():
        np.save(weights_dir / f"{key}.npy", array)
        print(f"  保存: {key}.npy ({array.shape})")
    
    result = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": 256,
        "weights_dir": str(weights_dir),
        "weight_keys": list(weights.keys())
    }
    
    result_path = weights_dir / "adapter_analysis.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 分析结果已保存: {result_path}")
    return result

def generate_fact_vectors():
    """为所有事实生成向量"""
    print(f"\n=== 步骤2：事实向量生成 ===")
    
    # 加载模型
    print("加载模型和适配器...")
    
    model_path = "/root/千问白盒化实验/models/Qwen2.5-0.5B-Instruct"
    adapter_path = "/root/千问白盒化实验/models/adapter/adapter_model.pt"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✅ tokenizer加载成功")
    except Exception as e:
        print(f"❌ tokenizer加载失败: {e}")
        return None
    
    # 加载适配器 - 修复键名问题
    try:
        checkpoint = torch.load(adapter_path, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(adapter_path, map_location='cpu')
    
    adapter_state_dict = checkpoint['model_state_dict']
    
    # 修复键名：network.0.weight -> 0.weight 等
    fixed_state_dict = {}
    for key, value in adapter_state_dict.items():
        # 转换 network.0.weight -> 0.weight
        if key.startswith("network."):
            new_key = key.replace("network.", "")
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value
    
    from torch import nn
    adapter = nn.Sequential(
        nn.Linear(896, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.LayerNorm(128)
    )
    
    adapter.load_state_dict(fixed_state_dict)
    adapter.eval()
    print("✅ 适配器加载成功（已修复键名）")
    
    # 加载事实数据
    print("\n加载事实数据...")
    data_path = "/root/千问白盒化实验/data/facts_dataset.json"
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        training_pairs = data.get("training_pairs", [])
        print(f"✅ 加载 {len(training_pairs)} 个训练对")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None
    
    # 提取所有事实文本
    fact_texts = []
    fact_ids = []
    
    for pair in training_pairs:
        fact_id = pair.get("fact_id", "unknown")
        fact_text = pair.get("fact_text", "")
        variations = pair.get("variations", [])
        
        # 添加主事实
        fact_texts.append(fact_text)
        fact_ids.append(fact_id)
        
        # 添加变体
        for variation in variations:
            fact_texts.append(variation)
            fact_ids.append(fact_id)
    
    print(f"总共 {len(fact_texts)} 个文本")
    
    # 生成向量
    print(f"\n生成向量...")
    vectors = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    adapter = adapter.to(device)
    
    for i, text in enumerate(fact_texts):
        if i % 20 == 0:
            print(f"  处理 {i+1}/{len(fact_texts)}")
        
        try:
            # 提取hidden_state（平均池化）
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1].mean(dim=1)
            
            # 通过适配器
            with torch.no_grad():
                vector = adapter(hidden_state)
            
            vectors.append(vector.cpu().numpy())
            
        except Exception as e:
            print(f"❌ 处理失败 '{text[:30]}...': {e}")
            vectors.append(np.zeros((1, 128)))
    
    # 保存结果
    print(f"\n保存结果...")
    
    vectors_dir = Path("/root/千问白盒化实验/vectors")
    vectors_dir.mkdir(parents=True, exist_ok=True)
    
    vectors_array = np.vstack(vectors)
    vectors_path = vectors_dir / "fact_vectors.npy"
    np.save(vectors_path, vectors_array)
    print(f"✅ 向量保存: {vectors_path} ({vectors_array.shape})")
    
    metadata_path = vectors_dir / "fact_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "fact_ids": fact_ids,
            "texts": fact_texts,
            "vector_shape": vectors_array.shape,
            "generation_time": time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 元数据保存: {metadata_path}")
    
    # 验证向量质量
    print(f"\n=== 向量质量验证 ===")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    same_fact_sims = []
    diff_fact_sims = []
    
    for i in range(len(fact_ids)):
        for j in range(i+1, min(len(fact_ids), i+50)):  # 限制比较数量
            sim = cosine_similarity(vectors_array[i:i+1], vectors_array[j:j+1])[0][0]
            if fact_ids[i] == fact_ids[j]:
                same_fact_sims.append(sim)
            else:
                diff_fact_sims.append(sim)
    
    if same_fact_sims and diff_fact_sims:
        avg_same = np.mean(same_fact_sims)
        avg_diff = np.mean(diff_fact_sims)
        discriminative = avg_same - avg_diff
        
        print(f"相同事实平均相似度: {avg_same:.4f}")
        print(f"不同事实平均相似度: {avg_diff:.4f}")
        print(f"判别性分数: {discriminative:.4f}")
        
        validation_result = {
            "average_same_sim": float(avg_same),
            "average_diff_sim": float(avg_diff),
            "discriminative_score": float(discriminative),
            "same_fact_count": len(same_fact_sims),
            "diff_fact_count": len(diff_fact_sims)
        }
        
        validation_path = vectors_dir / "vector_validation.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_result, f, indent=2)
        
        print(f"✅ 验证结果保存: {validation_path}")
    
    return {
        "vectors_path": str(vectors_path),
        "metadata_path": str(metadata_path),
        "vectors_shape": vectors_array.shape,
        "total_vectors": len(vectors)
    }

def main():
    print("=== 阶段二：步骤1-2执行 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1：分析适配器
    adapter_result = analyze_adapter()
    
    # 步骤2：生成事实向量
    vectors_result = generate_fact_vectors()
    
    print(f"\n=== 完成 ===")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if adapter_result and vectors_result:
        print("✅ 步骤1-2全部完成！")
        print(f"适配器权重目录: {adapter_result['weights_dir']}")
        print(f"事实向量文件: {vectors_result['vectors_path']}")
        print(f"向量形状: {vectors_result['vectors_shape']}")
        print(f"总向量数: {vectors_result['total_vectors']}")
        
        # 下一步建议
        print(f"\n=== 下一步建议 ===")
        print("1. 构建FAISS索引: python step2_faiss_index.py")
        print("2. 测试检索速度: python step3_retrieval_test.py")
        print("3. 创建可编辑性演示: python step4_edit_demo.py")
    else:
        print("❌ 部分步骤失败")

if __name__ == "__main__":
    main()