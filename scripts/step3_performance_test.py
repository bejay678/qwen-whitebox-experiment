import numpy as np
import faiss
import torch
import time
import json
from pathlib import Path

def load_resources():
    """加载所有资源"""
    print("=== 加载资源 ===")
    
    # 1. 加载FAISS索引
    index_path = "/root/千问白盒化实验/indices/faiss_index.bin"
    index = faiss.read_index(str(index_path))
    print(f"✅ FAISS索引加载: {index.ntotal} 个向量")
    
    # 2. 加载PyTorch基线
    baseline_path = "/root/千问白盒化实验/baseline/vectors_baseline.pt"
    vectors_tensor = torch.load(baseline_path)
    print(f"✅ PyTorch基线加载: {vectors_tensor.shape}")
    
    # 3. 加载测试查询
    vectors_path = "/root/千问白盒化实验/vectors/fact_vectors.npy"
    vectors = np.load(vectors_path)
    
    # 使用前20个向量作为测试查询
    test_queries = vectors[:20]
    faiss.normalize_L2(test_queries)
    
    print(f"✅ 测试查询: {len(test_queries)} 个")
    
    return index, vectors_tensor, test_queries

def pytorch_brute_force_search(query_vector, vectors_tensor, k=5):
    """PyTorch暴力检索"""
    # 归一化
    query_norm = query_vector / query_vector.norm(dim=1, keepdim=True)
    vectors_norm = vectors_tensor / vectors_tensor.norm(dim=1, keepdim=True)
    
    # 计算相似度
    similarities = torch.mm(query_norm, vectors_norm.T)
    
    # 获取top-k
    top_scores, top_indices = torch.topk(similarities, k, dim=1)
    
    return top_scores[0].numpy(), top_indices[0].numpy()

def faiss_search(query_vector, index, k=5):
    """FAISS检索"""
    query_vector = query_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)
    
    distances, indices = index.search(query_vector, k)
    
    return distances[0], indices[0]

def performance_test():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    # 加载资源
    index, vectors_tensor, test_queries = load_resources()
    
    # 测试配置
    k = 5  # top-5
    num_queries = len(test_queries)
    
    print(f"\n测试配置:")
    print(f"  查询数量: {num_queries}")
    print(f"  返回top-{k}")
    print(f"  向量总数: {index.ntotal}")
    
    # PyTorch测试
    print(f"\n=== PyTorch暴力检索测试 ===")
    pytorch_times = []
    pytorch_results = []
    
    for i, query in enumerate(test_queries):
        query_tensor = torch.FloatTensor(query).unsqueeze(0)
        
        start_time = time.perf_counter()
        scores, indices = pytorch_brute_force_search(query_tensor, vectors_tensor, k)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        pytorch_times.append(elapsed_ms)
        pytorch_results.append((scores, indices))
        
        if i < 3:  # 显示前3个结果
            print(f"  查询 {i+1}: {elapsed_ms:.2f} ms")
    
    pytorch_avg = np.mean(pytorch_times)
    pytorch_std = np.std(pytorch_times)
    print(f"  平均耗时: {pytorch_avg:.2f} ± {pytorch_std:.2f} ms")
    print(f"  总耗时: {sum(pytorch_times):.2f} ms")
    
    # FAISS测试
    print(f"\n=== FAISS检索测试 ===")
    faiss_times = []
    faiss_results = []
    
    for i, query in enumerate(test_queries):
        start_time = time.perf_counter()
        scores, indices = faiss_search(query, index, k)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        faiss_times.append(elapsed_ms)
        faiss_results.append((scores, indices))
        
        if i < 3:  # 显示前3个结果
            print(f"  查询 {i+1}: {elapsed_ms:.2f} ms")
    
    faiss_avg = np.mean(faiss_times)
    faiss_std = np.std(faiss_times)
    print(f"  平均耗时: {faiss_avg:.2f} ± {faiss_std:.2f} ms")
    print(f"  总耗时: {sum(faiss_times):.2f} ms")
    
    # 结果对比
    print(f"\n=== 性能对比 ===")
    speedup = pytorch_avg / faiss_avg if faiss_avg > 0 else 0
    
    print(f"PyTorch平均: {pytorch_avg:.2f} ms")
    print(f"FAISS平均: {faiss_avg:.2f} ms")
    print(f"加速比: {speedup:.2f}倍")
    
    # 准确性验证
    print(f"\n=== 准确性验证 ===")
    correct_matches = 0
    total_matches = 0
    
    for i in range(min(len(pytorch_results), len(faiss_results))):
        pytorch_indices = set(pytorch_results[i][1])
        faiss_indices = set(faiss_results[i][1])
        
        # 检查top-1是否一致
        if len(pytorch_indices) > 0 and len(faiss_indices) > 0:
            pytorch_top1 = pytorch_results[i][1][0]
            faiss_top1 = faiss_results[i][1][0]
            
            if pytorch_top1 == faiss_top1:
                correct_matches += 1
            total_matches += 1
    
    accuracy = correct_matches / total_matches if total_matches > 0 else 0
    print(f"Top-1一致性: {correct_matches}/{total_matches} ({accuracy:.1%})")
    
    # 保存结果
    print(f"\n=== 保存结果 ===")
    results_dir = Path("/root/千问白盒化实验/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "test_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "test_config": {
            "num_queries": num_queries,
            "top_k": k,
            "total_vectors": index.ntotal
        },
        "performance": {
            "pytorch": {
                "average_ms": float(pytorch_avg),
                "std_ms": float(pytorch_std),
                "total_ms": float(sum(pytorch_times)),
                "times_ms": [float(t) for t in pytorch_times]
            },
            "faiss": {
                "average_ms": float(faiss_avg),
                "std_ms": float(faiss_std),
                "total_ms": float(sum(faiss_times)),
                "times_ms": [float(t) for t in faiss_times]
            },
            "speedup": float(speedup)
        },
        "accuracy": {
            "correct_matches": correct_matches,
            "total_matches": total_matches,
            "top1_consistency": float(accuracy)
        }
    }
    
    result_path = results_dir / "performance_results.json"
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"✅ 性能结果已保存: {result_path}")
    
    # 生成报告
    print(f"\n=== 性能报告 ===")
    print(f"测试时间: {result_data['test_time']}")
    print(f"测试规模: {num_queries}个查询，{index.ntotal}个向量")
    print(f"\n性能指标:")
    print(f"  PyTorch暴力检索: {pytorch_avg:.2f} ± {pytorch_std:.2f} ms/查询")
    print(f"  FAISS IVF检索: {faiss_avg:.2f} ± {faiss_std:.2f} ms/查询")
    print(f"  加速比: {speedup:.2f}倍")
    print(f"  Top-1一致性: {accuracy:.1%}")
    
    # 建议
    print(f"\n=== 建议 ===")
    if speedup > 10:
        print("🎉 优秀！FAISS比PyTorch快10倍以上")
        print("   建议：可以继续优化索引参数")
    elif speedup > 5:
        print("✅ 良好！FAISS有明显加速效果")
        print("   建议：适合生产环境使用")
    elif speedup > 2:
        print("⚠️ 一般！FAISS有加速但不够显著")
        print("   建议：考虑增加向量数量或调整索引参数")
    else:
        print("❌ 较差！FAISS加速不明显")
        print("   建议：向量数量太少，需要更多数据")
    
    return result_data

def main():
    print("=== 阶段二：步骤5 - 性能对比测试 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = performance_test()
    
    print(f"\n=== 完成 ===")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if result:
        print("✅ 性能测试完成！")
        print(f"\n=== 下一步建议 ===")
        print("1. 可编辑性演示: python step4_edit_demo.py")
        print("2. 适配器C代码固化: python step5_c_adapter.py")
        print("3. 端到端集成: python step6_end_to_end.py")

if __name__ == "__main__":
    main()