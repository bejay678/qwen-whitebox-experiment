import numpy as np
import faiss
import json
import time
from pathlib import Path

class EditableMemorySystem:
    """可编辑记忆系统"""
    
    def __init__(self):
        print("=== 初始化可编辑记忆系统 ===")
        self.load_resources()
        self.create_editable_index()
        print(f"✅ 系统初始化完成")
        print(f"   向量数量: {self.index.ntotal}")
    
    def load_resources(self):
        """加载现有资源"""
        vectors_path = "/root/千问白盒化实验/vectors/fact_vectors.npy"
        self.vectors = np.load(vectors_path)
        faiss.normalize_L2(self.vectors)
        
        metadata_path = "/root/千问白盒化实验/vectors/fact_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.fact_ids = metadata.get("fact_ids", [])
        self.fact_texts = metadata.get("texts", [])
        
        print(f"✅ 加载 {len(self.fact_texts)} 个事实")
    
    def create_editable_index(self):
        """创建可编辑索引"""
        dimension = self.vectors.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)
        base_index = faiss.IndexIVFFlat(quantizer, dimension, 25, faiss.METRIC_INNER_PRODUCT)
        base_index.train(self.vectors)
        self.index = faiss.IndexIDMap(base_index)
        ids = np.arange(len(self.vectors), dtype=np.int64)
        self.index.add_with_ids(self.vectors, ids)
        print(f"✅ 创建可编辑索引，维度: {dimension}")
    
    def search(self, query_vector, k=5):
        """检索"""
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    "index": int(idx),
                    "similarity": float(distances[0][i]),
                    "text": self.fact_texts[idx] if idx < len(self.fact_texts) else "未知"
                })
        return results
    
    def add_fact(self, fact_text, vector=None):
        """添加新事实"""
        if vector is None:
            dimension = self.vectors.shape[1]
            vector = np.random.randn(1, dimension).astype(np.float32)
            faiss.normalize_L2(vector)
        else:
            vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
        
        new_id = self.index.ntotal
        self.index.add_with_ids(vector, np.array([new_id], dtype=np.int64))
        self.fact_ids.append(f"new_fact_{new_id}")
        self.fact_texts.append(fact_text)
        
        print(f"✅ 添加事实 [{new_id}]: {fact_text[:50]}...")
        return new_id
    
    def delete_fact(self, index):
        """删除事实"""
        if index >= len(self.fact_texts):
            print(f"❌ 索引 {index} 超出范围")
            return False
        
        print(f"🗑️  删除事实 [{index}]: {self.fact_texts[index][:50]}...")
        self.fact_texts[index] = f"[已删除] {self.fact_texts[index]}"
        return True
    
    def update_fact(self, index, new_text, new_vector=None):
        """更新事实"""
        if index >= len(self.fact_texts):
            print(f"❌ 索引 {index} 超出范围")
            return False
        
        print(f"🔄 更新事实 [{index}]: {self.fact_texts[index][:50]}... → {new_text[:50]}...")
        self.delete_fact(index)
        new_id = self.add_fact(new_text, new_vector)
        return new_id
    
    def list_facts(self, limit=10):
        """列出所有事实"""
        print(f"\n=== 事实列表（显示前{limit}个） ===")
        for i in range(min(limit, len(self.fact_texts))):
            status = "✅" if not self.fact_texts[i].startswith("[已删除]") else "❌"
            print(f"{status} [{i}] {self.fact_texts[i][:60]}...")

def run_demo():
    """运行演示"""
    print("=== 可编辑性演示 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建系统
    memory = EditableMemorySystem()
    
    print("\n" + "="*60)
    print("🎬 预设演示开始")
    print("="*60)
    
    # 1. 初始状态
    print("\n1. 📋 初始状态")
    memory.list_facts(5)
    
    # 2. 检索演示
    print("\n2. 🔍 检索演示")
    if len(memory.vectors) > 0:
        query_vector = memory.vectors[0:1]
        results = memory.search(query_vector, k=3)
        print(f"   查询: {memory.fact_texts[0][:50]}...")
        print(f"   找到 {len(results)} 个结果")
        for i, result in enumerate(results[:2]):
            print(f"     {i+1}. {result['text'][:50]}... (相似度: {result['similarity']:.4f})")
    
    # 3. 添加演示
    print("\n3. ➕ 添加新事实")
    new_facts = ["东京是日本首都", "北京是中国首都", "华盛顿是美国首都"]
    for fact in new_facts:
        dimension = memory.vectors.shape[1]
        new_vector = np.random.randn(1, dimension).astype(np.float32)
        faiss.normalize_L2(new_vector)
        memory.add_fact(fact, new_vector)
    
    # 4. 删除演示
    print("\n4. 🗑️  删除事实")
    if len(memory.fact_texts) > 2:
        memory.delete_fact(1)
    
    # 5. 更新演示
    print("\n5. 🔄 更新事实")
    if len(memory.fact_texts) > 3:
        old_text = memory.fact_texts[2]
        new_text = f"{old_text} - 更新版本"
        dimension = memory.vectors.shape[1]
        new_vector = np.random.randn(1, dimension).astype(np.float32)
        faiss.normalize_L2(new_vector)
        memory.update_fact(2, new_text, new_vector)
    
    # 6. 最终状态
    print("\n6. 📊 最终状态")
    memory.list_facts(8)
    print(f"   总事实数: {len(memory.fact_texts)}")
    print(f"   向量总数: {memory.index.ntotal}")
    
    # 保存状态
    print("\n💾 保存状态...")
    state_dir = Path("/root/千问白盒化实验/editable_state")
    state_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = state_dir / "editable_index.bin"
    faiss.write_index(memory.index, str(index_path))
    
    metadata = {
        "fact_texts": memory.fact_texts,
        "save_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_vectors": memory.index.ntotal
    }
    
    metadata_path = state_dir / "editable_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 状态已保存:")
    print(f"   索引: {index_path}")
    print(f"   元数据: {metadata_path}")
    
    print(f"\n=== 演示完成 ===")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_demo()