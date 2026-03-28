import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def analyze_adapter():
    """分析适配器结构和权重"""
    print("=== 步骤1：适配器分析 ===")
    
    # 1. 加载适配器
    adapter_path = "/root/千问白盒化实验/models/adapter/adapter_model.pt"
    
    try:
        checkpoint = torch.load(adapter_path, map_location='cpu')
        print(f"✅ 适配器加载成功")
    except Exception as e:
        print(f"❌ 适配器加载失败: {e}")
        return None
    
    # 2. 分析结构
    print(f"\n=== 适配器结构 ===")
    model_state_dict = checkpoint['model_state_dict']
    config = checkpoint['config']
    
    print(f"配置: {config}")
    print(f"\n权重键名:")
    for key in model_state_dict.keys():
        shape = model_state_dict[key].shape
        print(f"  {key}: {shape}")
    
    # 3. 提取权重
    weights = {}
    for key, tensor in model_state_dict.items():
        weights[key] = tensor.numpy()
    
    # 4. 计算参数数量
    total_params = sum(p.numel() for p in model_state_dict.values())
    print(f"\n=== 参数统计 ===")
    print(f"总参数: {total_params:,}")
    print(f"各层参数:")
    
    layer_params = {}
    for key, tensor in model_state_dict.items():
        layer_params[key] = tensor.numel()
        print(f"  {key}: {tensor.numel():,} ({tensor.shape})")
    
    # 5. 确定输入输出维度
    # 从权重形状推断
    input_dim = None
    output_dim = None
    
    for key, tensor in model_state_dict.items():
        if 'weight' in key and len(tensor.shape) == 2:
            if input_dim is None:
                input_dim = tensor.shape[1]  # 输入维度
            output_dim = tensor.shape[0]     # 输出维度
    
    print(f"\n=== 维度推断 ===")
    print(f"输入维度: {input_dim} (应为896)")
    print(f"输出维度: {output_dim} (应为128)")
    
    # 6. 保存权重为C代码可用格式
    print(f"\n=== 权重导出 ===")
    
    # 创建权重目录
    weights_dir = Path("/root/千问白盒化实验/models/adapter/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为numpy文件
    for key, array in weights.items():
        np.save(weights_dir / f"{key}.npy", array)
        print(f"  保存: {key}.npy ({array.shape})")
    
    # 保存为文本文件（C代码使用）
    for key, array in weights.items():
        txt_path = weights_dir / f"{key}.txt"
        with open(txt_path, 'w') as f:
            # 扁平化并写入
            flat = array.flatten()
            for i, val in enumerate(flat):
                f.write(f"{val:.6f}")
                if i < len(flat) - 1:
                    f.write(", ")
                if (i + 1) % 10 == 0:
                    f.write("\n")
        print(f"  文本格式: {key}.txt ({len(flat)}个值)")
    
    # 7. 生成C代码模板
    print(f"\n=== C代码模板 ===")
    
    c_template = f"""
// adapter_module.c - 适配器C语言实现
// 输入维度: {input_dim}
// 输出维度: {output_dim}
// 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_DIM {input_dim}
#define HIDDEN_DIM 256
#define OUTPUT_DIM {output_dim}

// 权重和偏置（从训练好的适配器加载）
extern float fc1_weight[HIDDEN_DIM][INPUT_DIM];
extern float fc1_bias[HIDDEN_DIM];
extern float fc2_weight[OUTPUT_DIM][HIDDEN_DIM];
extern float fc2_bias[OUTPUT_DIM];

// ReLU激活函数
static inline float relu(float x) {{
    return x > 0 ? x : 0;
}}

// 适配器前向传播
void adapter_forward(float* input, float* output) {{
    float hidden[HIDDEN_DIM];
    
    // 第一层: input_dim -> hidden_dim
    for (int i = 0; i < HIDDEN_DIM; i++) {{
        hidden[i] = fc1_bias[i];
        for (int j = 0; j < INPUT_DIM; j++) {{
            hidden[i] += input[j] * fc1_weight[i][j];
        }}
        hidden[i] = relu(hidden[i]);
    }}
    
    // 第二层: hidden_dim -> output_dim
    for (int i = 0; i < OUTPUT_DIM; i++) {{
        output[i] = fc2_bias[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {{
            output[i] += hidden[j] * fc2_weight[i][j];
        }}
    }}
}}

// 测试函数
void test_adapter() {{
    float input[INPUT_DIM];
    float output[OUTPUT_DIM];
    
    // 初始化输入（示例）
    for (int i = 0; i < INPUT_DIM; i++) {{
        input[i] = 0.1f;
    }}
    
    // 运行适配器
    adapter_forward(input, output);
    
    printf("适配器测试完成\\n");
    printf("输入维度: %d\\n", INPUT_DIM);
    printf("输出维度: %d\\n", OUTPUT_DIM);
    printf("输出前5个值: ");
    for (int i = 0; i < 5 && i < OUTPUT_DIM; i++) {{
        printf("%.6f ", output[i]);
    }}
    printf("\\n");
}}
"""
    
    c_path = weights_dir / "adapter_module.c"
    with open(c_path, 'w') as f:
        f.write(c_template)
    
    print(f"✅ C代码模板已生成: {c_path}")
    
    # 8. 返回分析结果
    result = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": 256,
        "total_params": total_params,
        "layer_params": layer_params,
        "weights_dir": str(weights_dir),
        "c_template_path": str(c_path)
    }
    
    # 保存分析结果
    result_path = weights_dir / "adapter_analysis.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 分析结果已保存: {result_path}")
    
    return result

def generate_fact_vectors():
    """为所有事实生成向量"""
    print(f"\n=== 步骤2：事实向量生成 ===")
    
    # 1. 加载模型和适配器
    print("加载模型和适配器...")
    
    model_path = "/root/千问白盒化实验/models/Qwen2.5-0.5B-Instruct"
    adapter_path = "/root/千问白盒化实验/models/adapter/adapter_model.pt"
    
    try:
        # 加载模型
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
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✅ tokenizer加载成功")
    except Exception as e:
        print(f"❌ tokenizer加载失败: {e}")
        return None
    
    try:
        # 加载适配器
        checkpoint = torch.load(adapter_path, map_location='cpu')
        adapter_state_dict = checkpoint['model_state_dict']
        
        # 重建适配器
        from torch import nn
        adapter = nn.Sequential(
            nn.Linear(896, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        # 加载权重
        adapter.load_state_dict(adapter_state_dict)
        adapter.eval()
        print("✅ 适配器加载成功")
    except Exception as e:
        print(f"❌ 适配器加载失败: {e}")
        return None
    
    # 2. 加载事实数据
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
    
    # 3. 提取所有事实文本
    fact_texts = []
    fact_ids = []
    metadata = []
    
    for pair in training_pairs:
        fact_id = pair.get("fact_id", "unknown")
        fact_text = pair.get("fact_text", "")
        variations = pair.get("variations", [])
        
        # 添加主事实
        fact_texts.append(fact_text)
        fact_ids.append(fact_id)
        metadata.append({
            "fact_id": fact_id,
            "text": fact_text,
            "type": "main",
            "variations": variations
        })
        
        # 添加变体
        for i, variation in enumerate(variations):
            fact_texts.append(variation)
            fact_ids.append(fact_id)
            metadata.append({
                "fact_id": fact_id,
                "text": variation,
                "type": f"variation_{i+1}",
                "original": fact_text
            })
    
    print(f"总共 {len(fact_texts)} 个文本（{len(set(fact_ids))} 个不同事实）")
    
    # 4. 生成向量
    print(f"\n生成向量...")
    vectors = []
    successful = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    adapter = adapter.to(device)
    
    for i, text in enumerate(fact_texts):
        if i % 20 == 0:
            print(f"  处理 {i+1}/{len(fact_texts)}: '{text[:30]}...'")
        
        try:
            # 提取hidden_state（平均池化）
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1].mean(dim=1)  # 平均池化
            
            # 通过适配器
            with torch.no_grad():
                vector = adapter(hidden_state)
            
            vectors.append(vector.cpu().numpy())
            successful += 1
            
        except Exception as e:
            print(f"❌ 处理失败 '{text[:30]}...': {e}")
            vectors.append(np.zeros((1, 128)))  # 填充零向量
    
    print(f"✅ 成功生成 {successful}/{len(fact_texts)} 个向量")
    
    # 5. 保存向量和元数据
    print(f"\n保存结果...")
    
    vectors_dir = Path("/root/千问白盒化实验/vectors")
    vectors_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存向量为numpy数组
    vectors_array = np.vstack(vectors)
    vectors_path = vectors_dir / "fact_vectors.npy"
    np.save(vectors_path, vectors_array)
    print(f"✅ 向量保存: {vectors_path} ({vectors_array.shape})")
    
    # 保存元数据
    metadata_path = vectors_dir / "fact_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "fact_ids": fact_ids,
            "texts": fact_texts,
            "metadata": metadata,
            "vector_shape": vectors_array.shape,
            "generation_time": time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 元数据保存: {metadata_path}")
    
    # 6. 验证向量质量
    print(f"\n=== 向量质量验证 ===")
    
    # 计算相同事实和不同事实的相似度
    from sklearn.metrics.pairwise import cosine_similarity
    
    same_fact_sims = []
    diff_fact_sims = []
    
    for i in range(len(fact_ids)):
        for j in range(i+1, len(fact_ids)):
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
        
        # 保存验证结果
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
        "successful_count": successful,
        "total_count": len(fact_texts)
    }

def main():
    print("=== 阶段二：步骤1-2执行 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1：分析适配器
    adapter_result = analyze_adapter()
    if adapter_result is None:
        print("❌ 适配器分析失败")
        return
    
    # 步骤2：生成事实向量
    vectors_result = generate_fact_vectors()
    if vectors