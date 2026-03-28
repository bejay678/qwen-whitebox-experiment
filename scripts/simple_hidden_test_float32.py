import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys

def load_model():
    """加载模型和tokenizer - 使用float32"""
    model_path = "/root/千问白盒化实验/models/Qwen2.5-0.5B-Instruct"
    
    print("加载模型...")
    try:
        # 使用float32而不是bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # 改为float32
            trust_remote_code=True
        )
        # 手动移动到GPU
        model = model.to("cuda")
        print(f"✅ 模型加载成功，设备: {model.device}")
        print(f"   模型dtype: {model.dtype}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None
    
    print("加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✅ tokenizer加载成功")
    except Exception as e:
        print(f"❌ tokenizer加载失败: {e}")
        return model, None
    
    return model, tokenizer

def get_hidden_state(model, tokenizer, text):
    """提取文本的hidden_state"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # 取最后一层的第一个token的hidden_state
            hidden_state = outputs.hidden_states[-1][:, 0, :]
            
            # 转换为float32 numpy数组
            hidden_state = hidden_state.float().cpu().numpy()
        
        return hidden_state
    except Exception as e:
        print(f"❌ 提取hidden_state失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=== 千问2.5-0.5B hidden_state判别性测试 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 检查环境
    print(f"\n=== 环境检查 ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 2. 加载模型
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("❌ 模型加载失败，退出测试")
        return
    
    # 3. 测试数据 - 简化版本，先测试少量
    test_pairs = [
        ("北京是中国的首都", "中国的首都是北京"),
        ("北京是中国的首都", "上海是中国最大的城市"),
    ]
    
    # 4. 提取hidden_state
    print(f"\n=== 提取hidden_state ===")
    hidden_states = []
    texts = []
    
    all_texts = [pair[0] for pair in test_pairs] + [pair[1] for pair in test_pairs]
    for i, text in enumerate(all_texts):
        print(f"  处理文本 {i+1}/{len(all_texts)}: '{text}'")
        hidden = get_hidden_state(model, tokenizer, text)
        if hidden is None:
            print("❌ hidden_state提取失败，退出测试")
            return
        hidden_states.append(hidden)
        texts.append(text)
        print(f"    hidden_state shape: {hidden.shape}")
    
    # 5. 计算相似度
    print(f"\n=== 相似度计算 ===")
    hidden_array = np.vstack(hidden_states)
    sim_matrix = cosine_similarity(hidden_array)
    
    print("\n相似度矩阵:")
    for i in range(len(texts)):
        row = [f"{sim_matrix[i, j]:.4f}" for j in range(len(texts))]
        print(f"  {texts[i][:15]:15} {row}")
    
    # 6. 分析结果
    print(f"\n=== 结果分析 ===")
    pair_count = len(test_pairs)
    results = []
    
    for i in range(pair_count):
        same_fact_sim = sim_matrix[i, i + pair_count]
        
        # 找不同的对比
        diff_idx = (i + 1) % pair_count + pair_count
        diff_fact_sim = sim_matrix[i, diff_idx]
        
        result = {
            "pair_id": i+1,
            "text1": texts[i],
            "text2": texts[i + pair_count],
            "same_fact_similarity": float(same_fact_sim),
            "diff_fact_similarity": float(diff_fact_sim),
            "discriminative_score": float(same_fact_sim - diff_fact_sim)
        }
        results.append(result)
        
        print(f"\n测试对 {i+1}:")
        print(f"  相同事实: '{texts[i]}'")
        print(f"           '{texts[i+pair_count]}'")
        print(f"  相似度: {same_fact_sim:.4f}")
        print(f"  不同事实相似度: {diff_fact_sim:.4f}")
        print(f"  判别性分数: {same_fact_sim - diff_fact_sim:.4f}")
    
    # 7. 总结
    print(f"\n=== 测试总结 ===")
    if results:
        same_sims = [r["same_fact_similarity"] for r in results]
        diff_sims = [r["diff_fact_similarity"] for r in results]
        disc_scores = [r["discriminative_score"] for r in results]
        
        print(f"测试对数: {len(results)}")
        print(f"相同事实平均相似度: {np.mean(same_sims):.4f}")
        print(f"不同事实平均相似度: {np.mean(diff_sims):.4f}")
        print(f"平均判别性分数: {np.mean(disc_scores):.4f}")
        
        # 8. 保存结果
        final_result = {
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Qwen2.5-0.5B-Instruct",
            "environment": {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                "model_dtype": str(model.dtype)
            },
            "test_pairs": test_pairs,
            "detailed_results": results,
            "summary": {
                "average_same_sim": float(np.mean(same_sims)),
                "average_diff_sim": float(np.mean(diff_sims)),
                "average_discriminative": float(np.mean(disc_scores))
            }
        }
        
        output_file = "/root/千问白盒化实验/logs/phase1_results.json"
        with open(output_file, "w") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 结果已保存到: {output_file}")
        
        # 9. 决策建议
        avg_disc = final_result["summary"]["average_discriminative"]
        print(f"\n=== 决策建议 ===")
        if avg_disc > 0.4:
            print("🎉 优秀！hidden_state判别性很强 (>0.4)")
            print("   建议：立即开始阶段二实验（FAISS固化）")
        elif avg_disc > 0.2:
            print("✅ 良好！hidden_state有足够判别性 (0.2-0.4)")
            print("   建议：可以继续实验，考虑优化适配器设计")
        elif avg_disc > 0.1:
            print("⚠️ 一般！hidden_state判别性较弱 (0.1-0.2)")
            print("   建议：需要调整模型或训练策略")
        else:
            print("❌ 较差！hidden_state判别性不足 (<0.1)")
            print("   建议：重新设计实验方案")
    else:
        print("❌ 没有有效结果")
    
    print(f"\n测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()