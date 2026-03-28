import numpy as np
import faiss
import json
import time
from pathlib import Path

def build_faiss_index():
    """构建FAISS索引"""
    print("=== 步骤3：构建FAISS索引 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载向量
    vectors_path = "/root/千问白盒化实验/vectors/fact_vectors.npy"
    
    try:
        vectors = np.load(vectors_path)
        print(f"✅ 加载向量: {vectors.shape}")
    except Exception as e:
        print(f"❌ 加载向量失败: {e}")
        return None
    
    # 2. 归一化向量（用于余弦相似度）
    print("归一化向量...")
    faiss.normalize_L2(vectors)
    
    # 3. 构建FAISS索引
    print("\n=== 构建索引 ===")
    dimension = vectors.shape[1]
    nlist = min(100, vectors.shape[0] // 10)  # 聚类中心数
    
    print(f"向量维度: {dimension}")
    print(f"向量数量: {vectors.shape[0]}")
    print(f"聚类中心数: {nlist}")
    
    # 创建量化器
    quantizer = faiss.IndexFlatIP(dimension)  # 内积=余弦相似度（已归一化）
    
    # 创建IVF索引
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # 4. 训练索引
    print("训练索引...")
    index.train(vectors)
    
    # 5. 添加向量
    print("添加向量到索引...")
    index.add(vectors)
    
    # 6. 保存索引
    print("\n=== 保存索引 ===")
    index_dir = Path("/root/千问白盒化实验/indices")
    index_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = index_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS索引已保存: {index_path}")
    
    # 7. 创建简单检索测试
    print("\n=== 简单检索测试 ===")
    
    # 测试查询
    test_queries = vectors[:5]  # 用前5个向量作为查询
    k = 3  # 返回top-3
    
    print(f"测试 {len(test_queries)} 个查询，返回top-{k}")
    
    for i, query in enumerate(test_queries):
        query = query.reshape(1, -1)
        faiss.normalize_L2(query)
        
        distances, indices = index.search(query, k)
        
        print(f"\n查询 {i+1}:")
        print(f"  最近邻索引: {indices[0]}")
        print(f"  相似度: {distances[0]}")
        
        # 加载元数据查看文本
        metadata_path = "/root/千问白盒化实验/vectors/fact_metadata.json"
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            texts = metadata.get("texts", [])
            if texts:
                print(f"  对应文本:")
                for idx in indices[0]:
                    if idx < len(texts):
                        print(f"    [{idx}] {texts[idx][:50]}...")
        except:
            pass
    
    # 8. 保存索引配置
    config = {
        "dimension": dimension,
        "num_vectors": vectors.shape[0],
        "nlist": nlist,
        "metric": "METRIC_INNER_PRODUCT (余弦相似度)",
        "index_type": "IndexIVFFlat",
        "build_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "index_path": str(index_path),
        "vectors_path": str(vectors_path)
    }
    
    config_path = index_dir / "index_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ 索引配置已保存: {config_path}")
    
    return {
        "index_path": str(index_path),
        "config_path": str(config_path),
        "dimension": dimension,
        "num_vectors": vectors.shape[0],
        "nlist": nlist
    }

def create_pytorch_baseline():
    """创建PyTorch暴力检索基线"""
    print(f"\n=== 步骤4：创建PyTorch基线 ===")
    
    import torch
    
    # 加载向量
    vectors_path = "/root/千问白盒化实验/vectors/fact_vectors.npy"
    vectors = np.load(vectors_path)
    
    # 转换为PyTorch张量
    vectors_tensor = torch.FloatTensor(vectors)
    
    # 保存基线
    baseline_dir = Path("/root/千问白盒化实验/baseline")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为PyTorch格式
    baseline_path = baseline_dir / "vectors_baseline.pt"
    torch.save(vectors_tensor, baseline_path)
    
    print(f"✅ PyTorch基线已保存: {baseline_path}")
    print(f"   向量形状: {vectors_tensor.shape}")
    
    # 创建简单的暴力检索函数
    baseline_code = '''
import torch
import numpy as np
import time

def pytorch_brute_force_search(query_vector, vectors_tensor, k=5):
    """
    PyTorch暴力检索
    query_vector: (1, 128) 查询向量
    vectors_tensor: (n, 128) 所有向量
    k: 返回top-k
    """
    # 计算余弦相似度
    query_norm = query_vector / query_vector.norm(dim=1, keepdim=True)
    vectors_norm = vectors_tensor / vectors_tensor.norm(dim=1, keepdim=True)
    
    similarities = torch.mm(query_norm, vectors_norm.T)  # (1, n)
    
    # 获取top-k
    top_scores, top_indices = torch.topk(similarities, k, dim=1)
    
    return top_scores[0].numpy(), top_indices[0].numpy()

# 使用示例
if __name__ == "__main__":
    # 加载向量
    vectors = torch.load("/root/千问白盒化实验/baseline/vectors_baseline.pt")
    
    # 测试查询
    test_query = vectors[0:1]  # 用第一个向量作为查询
    
    start_time = time.time()
    scores, indices = pytorch_brute_force_search(test_query, vectors, k=3)
    end_time = time.time()
    
    print(f"PyTorch暴力检索结果:")
    print(f"  索引: {indices}")
    print(f"  相似度: {scores}")
    print(f"  耗时: {(end_time - start_time)*1000:.2f} ms")
'''
    
    code_path = baseline_dir / "pytorch_baseline.py"
    with open(code_path, 'w') as f:
        f.write(baseline_code)
    
    print(f"✅ PyTorch检索代码已保存: {code_path}")
    
    return {
        "baseline_path": str(baseline_path),
        "code_path": str(code_path),
        "num_vectors": vectors.shape[0],
        "dimension": vectors.shape[1]
    }

def main():
    print("=== 阶段二：步骤3-4执行 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤3：构建FAISS索引
    faiss_result = build_faiss_index()
    
    # 步骤4：创建PyTorch基线
    baseline_result = create_pytorch_baseline()
    
    print(f"\n=== 完成 ===")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if faiss_result and baseline_result:
        print("✅ 步骤3-4全部完成！")
        print(f"\n=== 生成的文件 ===")
        print(f"FAISS索引: {faiss_result['index_path']}")
        print(f"索引配置: {faiss_result['config_path']}")
        print(f"PyTorch基线: {baseline_result['baseline_path']}")
        print(f"检索代码: {baseline_result['code_path']}")
        
        print(f"\n=== 下一步建议 ===")
        print("1. 性能对比测试: python step3_performance_test.py")
        print("2. 可编辑性演示: python step4_edit_demo.py")
        print("3. 适配器C代码固化: python step5_c_adapter.py")
    else:
        print("❌ 部分步骤失败")

if __name__ == "__main__":
    main()