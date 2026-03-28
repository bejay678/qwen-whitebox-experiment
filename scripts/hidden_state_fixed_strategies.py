import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys

def load_model():
    """加载模型和tokenizer"""
    model_path = "/root/千问白盒化实验/models/Qwen2.5-0.5B-Instruct"
    
    print("加载模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to("cuda")
        print(f"✅ 模型加载成功，设备: {model.device}")
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

def extract_hidden_state_strategy1(model, tokenizer, text):
    """策略1: 第一个token的hidden_state"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1][:, 0, :].float().cpu().numpy()
    
    return hidden_state

def extract_hidden_state_strategy2(model, tokenizer, text):
    """策略2: 所有token的平均hidden_state"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1].mean(dim=1).float().cpu().numpy()
    
    return hidden_state

def extract_hidden_state_strategy3(model, tokenizer, text):
    """策略3: 中间层的hidden_state平均值"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 模型有24层，取中间层12
        middle_layer = len(outputs.hidden_states) // 2
        hidden_state = outputs.hidden_states[middle_layer].mean(dim=1).float().cpu().numpy()
    
    return hidden_state

def extract_hidden_state_strategy4_fixed(model, tokenizer, text):
    """策略4: 最后3层的简单平均（修复版）"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 取最后3层
        last_layers = outputs.hidden_states[-3:]
        # 简单平均
        avg_hidden = torch.stack(last_layers).mean(dim=0).mean(dim=1)
        hidden_state = avg_hidden.float().cpu().numpy()
    
    return hidden_state

def extract_hidden_state_strategy5_simple(model, tokenizer, text):
    """策略5: 最后层的CLS token（如果有）或第一个token"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 尝试找CLS token（通常是第一个）
        hidden_state = outputs.hidden_states[-1][:, 0, :].float().cpu().numpy()
    
    return hidden_state

def extract_hidden_state_strategy6(model, tokenizer, text):
    """策略6: 使用pooler_output（如果有）"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        # 如果没有pooler_output，使用最后一个token
        last_token_idx = inputs["input_ids"].shape[1] - 1
        hidden_state = outputs.hidden_states[-1][:, last_token_idx, :].float().cpu().numpy()
    
    return hidden_state

def test_strategy(strategy_func, strategy_name, model, tokenizer, test_pairs):
    """测试一个提取策略"""
    print(f"\n=== 测试策略: {strategy_name} ===")
    
    all_texts = [pair[0] for pair in test_pairs] + [pair[1] for pair in test_pairs]
    hidden_states = []
    
    for i, text in enumerate(all_texts):
        print(f"  处理文本 {i+1}/{len(all_texts)}: '{text}'")
        try:
            hidden = strategy_func(model, tokenizer, text)
            hidden_states.append(hidden)
        except Exception as e:
            print(f"❌ {strategy_name} 提取失败: {e}")
            return None
    
    # 计算相似度
    hidden_array = np.vstack(hidden_states)
    sim_matrix = cosine_similarity(hidden_array)
    
    # 分析结果
    pair_count = len(test_pairs)
    results = []
    
    for i in range(pair_count):
        same_fact_sim = sim_matrix[i, i + pair_count]
        diff_idx = (i + 1) % pair_count + pair_count
        diff_fact_sim = sim_matrix[i, diff_idx]
        
        results.append({
            "pair_id": i+1,
            "same_fact_similarity": float(same_fact_sim),
            "diff_fact_similarity": float(diff_fact_sim),
            "discriminative_score": float(same_fact_sim - diff_fact_sim)
        })
    
    # 计算统计
    same_sims = [r["same_fact_similarity"] for r in results]
    diff_sims = [r["diff_fact_similarity"] for r in results]
    disc_scores = [r["discriminative_score"] for r in results]
    
    avg_same = np.mean(same_sims)
    avg_diff = np.mean(diff_sims)
    avg_disc = np.mean(disc_scores)
    
    print(f"  相同事实平均相似度: {avg_same:.4f}")
    print(f"  不同事实平均相似度: {avg_diff:.4f}")
    print(f"  平均判别性分数: {avg_disc:.4f}")
    
    return {
        "strategy_name": strategy_name,
        "average_same_sim": float(avg_same),
        "average_diff_sim": float(avg_diff),
        "average_discriminative": float(avg_disc),
        "detailed_results": results
    }

def main():
    print("=== 千问2.5-0.5B hidden_state提取策略对比测试 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 环境检查
    print(f"\n=== 环境检查 ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 加载模型
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        return
    
    # 测试数据
    test_pairs = [
        ("北京是中国的首都", "中国的首都是北京"),
        ("北京是中国的首都", "上海是中国最大的城市"),
        ("水的化学式是H2O", "H2O是水的化学式"),
        ("水的化学式是H2O", "二氧化碳的化学式是CO2"),
    ]
    
    print(f"\n=== 测试数据 ===")
    for i, (text1, text2) in enumerate(test_pairs):
        print(f"  对{i+1}: '{text1}' vs '{text2}'")
    
    # 定义策略
    strategies = [
        (extract_hidden_state_strategy1, "策略1: 第一个token"),
        (extract_hidden_state_strategy2, "策略2: 所有token平均"),
        (extract_hidden_state_strategy3, "策略3: 中间层平均"),
        (extract_hidden_state_strategy4_fixed, "策略4: 最后3层平均"),
        (extract_hidden_state_strategy5_simple, "策略5: CLS token"),
        (extract_hidden_state_strategy6, "策略6: 最后一个token"),
    ]
    
    # 测试所有策略
    all_results = []
    for strategy_func, strategy_name in strategies:
        result = test_strategy(strategy_func, strategy_name, model, tokenizer, test_pairs)
        if result is not None:
            all_results.append(result)
    
    # 结果总结
    print(f"\n=== 策略对比总结 ===")
    print(f"{'策略':<20} {'相同事实相似度':<15} {'不同事实相似度':<15} {'判别性分数':<15} {'评价':<10}")
    print("-" * 80)
    
    best_strategy = None
    best_score = -float('inf')
    
    for result in all_results:
        avg_disc = result["average_discriminative"]
        
        # 评价
        if avg_disc > 0.4:
            evaluation = "🎉 优秀"
        elif avg_disc > 0.2:
            evaluation = "✅ 良好"
        elif avg_disc > 0.1:
            evaluation = "⚠️ 一般"
        elif avg_disc > 0:
            evaluation = "❌ 较弱"
        else:
            evaluation = "❌ 很差"
        
        print(f"{result['strategy_name']:<20} "
              f"{result['average_same_sim']:<15.4f} "
              f"{result['average_diff_sim']:<15.4f} "
              f"{avg_disc:<15.4f} "
              f"{evaluation:<10}")
        
        if avg_disc > best_score:
            best_score = avg_disc
            best_strategy = result
    
    # 保存结果
    final_result = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen2.5-0.5B-Instruct",
        "test_pairs": test_pairs,
        "strategies_results": all_results,
        "best_strategy": best_strategy
    }
    
    output_file = "/root/千问白盒化实验/logs/phase1_strategies_final.json"
    with open(output_file, "w") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 所有结果已保存到: {output_file}")
    
    # 详细分析
    print(f"\n=== 详细分析 ===")
    if best_strategy:
        print(f"最佳策略: {best_strategy['strategy_name']}")
        print(f"判别性分数: {best_strategy['average_discriminative']:.4f}")
        
        # 显示每个测试对的详细结果
        print(f"\n最佳策略的详细结果:")
        for result in best_strategy["detailed_results"]:
            pair = test_pairs[result["pair_id"]-1]
            print(f"  对{result['pair_id']}: '{pair[0]}' vs '{pair[1]}'")
            print(f"    相同事实相似度: {result['same_fact_similarity']:.4f}")
            print(f"    不同事实相似度: {result['diff_fact_similarity']:.4f}")
            print(f"    判别性分数: {result['discriminative_score']:.4f}")
    
    # 决策建议
    print(f"\n=== 最终决策建议 ===")
    if best_strategy:
        best_disc = best_strategy["average_discriminative"]
        
        if best_disc > 0.4:
            print("🎉 优秀！找到有效的hidden_state提取策略")
            print("   建议：立即使用此策略开始阶段二实验（FAISS固化）")
        elif best_disc > 0.2:
            print("✅ 良好！hidden_state有足够判别性")
            print("   建议：可以继续实验，使用策略2（所有token平均）")
        elif best_disc > 0.1:
            print("⚠️ 一般！判别性较弱但可用")
            print("   建议：考虑结合适配器训练来增强特征提取")
        elif best_disc > 0:
            print("❌ 较弱！判别性不足")
            print("   建议：必须训练适配器，原始hidden_state判别性不足")
        else:
            print("❌ 很差！所有策略都无效")
            print("   建议：直接开始适配器训练，跳过hidden_state验证")
    else:
        print("❌ 所有策略都失败")
        print("   建议：直接开始适配器训练")
    
    print(f"\n测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()