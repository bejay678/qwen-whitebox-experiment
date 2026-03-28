import numpy as np
import faiss
import json
import time
from pathlib import Path
import sys

class EditableMemorySystem:
    """可编辑记忆系统"""
    
    def __init__(self):
        print("=== 初始化可编辑记忆系统 ===")
        
        # 加载现有资源
        self.load_resources()
        
        # 创建可编辑的FAISS索引
        self.create_editable_index()
        
        print(f"✅ 系统初始化完成")
        print(f"   向量数量: {self.index.ntotal}")
        print(f"   事实数量: {len(self.fact_texts)}")
    
    def load_resources(self):
        """加载现有资源"""
        # 加载向量
        vectors_path = "/root/千问白盒化实验/vectors/fact_vectors.npy"
        self.vectors = np.load(vectors_path)
        faiss.normalize_L2(self.vectors)
        
        # 加载元数据
        metadata_path = "/root/千问白盒化实验/vectors/fact_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.fact_ids = metadata.get("fact_ids", [])
        self.fact_texts = metadata.get("texts", [])
        
        print(f"✅ 加载 {len(self.fact_texts)} 个事实")
    
    def create_editable_index(self):
        """创建可编辑索引"""
        dimension = self.vectors.shape[1]
        
        # 使用IndexIDMap包装基础索引，支持动态ID
        quantizer = faiss.IndexFlatIP(dimension)
        base_index = faiss.IndexIVFFlat(quantizer, dimension, 25, faiss.METRIC_INNER_PRODUCT)
        
        # 训练索引
        base_index.train(self.vectors)
        
        # 创建ID映射索引
        self.index = faiss.IndexIDMap(base_index)
        
        # 添加初始向量，使用连续ID
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
            if idx != -1:  # -1表示无效结果
                results.append({
                    "index": int(idx),
                    "similarity": float(distances[0][i]),
                    "text": self.fact_texts[idx] if idx < len(self.fact_texts) else "未知",
                    "fact_id": self.fact_ids[idx] if idx < len(self.fact_ids) else "未知"
                })
        
        return results
    
    def add_fact(self, fact_text, fact_id=None, vector=None):
        """添加新事实"""
        if vector is None:
            print("⚠️ 需要提供向量，暂时使用随机向量演示")
            # 实际应用中应该通过适配器生成向量
            dimension = self.vectors.shape[1]
            vector = np.random.randn(1, dimension).astype(np.float32)
            faiss.normalize_L2(vector)
        else:
            vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
        
        # 生成新ID
        new_id = self.index.ntotal
        
        # 添加到索引
        self.index.add_with_ids(vector, np.array([new_id], dtype=np.int64))
        
        # 更新元数据
        if fact_id is None:
            fact_id = f"new_fact_{new_id}"
        
        self.fact_ids.append(fact_id)
        self.fact_texts.append(fact_text)
        
        print(f"✅ 添加事实 [{new_id}]: {fact_text[:50]}...")
        print(f"   事实ID: {fact_id}")
        print(f"   当前向量总数: {self.index.ntotal}")
        
        return new_id
    
    def delete_fact(self, index):
        """删除事实（通过标记删除）"""
        if index >= len(self.fact_texts):
            print(f"❌ 索引 {index} 超出范围")
            return False
        
        # FAISS不支持直接删除，我们通过重建索引实现
        print(f"🗑️  删除事实 [{index}]: {self.fact_texts[index][:50]}...")
        
        # 标记为已删除
        self.fact_texts[index] = f"[已删除] {self.fact_texts[index]}"
        
        # 实际应用中应该重建索引，这里简化处理
        print(f"   注意：FAISS不支持直接删除，实际应用需重建索引")
        
        return True
    
    def update_fact(self, index, new_text, new_vector=None):
        """更新事实"""
        if index >= len(self.fact_texts):
            print(f"❌ 索引 {index} 超出范围")
            return False
        
        old_text = self.fact_texts[index]
        print(f"🔄 更新事实 [{index}]:")
        print(f"   原文本: {old_text[:50]}...")
        print(f"   新文本: {new_text[:50]}...")
        
        # 先删除旧事实
        self.delete_fact(index)
        
        # 添加新事实
        fact_id = self.fact_ids[index] if index < len(self.fact_ids) else f"fact_{index}"
        new_id = self.add_fact(new_text, fact_id, new_vector)
        
        return new_id
    
    def list_facts(self, limit=10):
        """列出所有事实"""
        print(f"\n=== 事实列表（显示前{limit}个） ===")
        
        for i in range(min(limit, len(self.fact_texts))):
            status = "✅" if not self.fact_texts[i].startswith("[已删除]") else "❌"
            print(f"{status} [{i}] {self.fact_texts[i][:60]}...")
        
        if len(self.fact_texts) > limit:
            print(f"... 还有 {len(self.fact_texts) - limit} 个事实")
    
    def interactive_demo(self):
        """交互式演示"""
        print("\n" + "="*60)
        print("🎭 可编辑性演示 - 交互模式")
        print("="*60)
        
        while True:
            print("\n=== 菜单 ===")
            print("1. 检索事实")
            print("2. 添加新事实")
            print("3. 删除事实")
            print("4. 更新事实")
            print("5. 列出事实")
            print("6. 运行预设演示")
            print("0. 退出")
            
            choice = input("\n请选择操作 (0-6): ").strip()
            
            if choice == "0":
                print("👋 退出演示")
                break
            
            elif choice == "1":
                self.demo_search()
            
            elif choice == "2":
                self.demo_add()
            
            elif choice == "3":
                self.demo_delete()
            
            elif choice == "4":
                self.demo_update()
            
            elif choice == "5":
                self.list_facts(20)
            
            elif choice == "6":
                self.run_preset_demo()
            
            else:
                print("❌ 无效选择")
    
    def demo_search(self):
        """演示检索"""
        print("\n=== 检索演示 ===")
        
        # 使用现有向量作为查询
        if len(self.vectors) > 0:
            query_idx = 0  # 使用第一个向量
            query_vector = self.vectors[query_idx:query_idx+1]
            
            print(f"查询: {self.fact_texts[query_idx][:50]}...")
            
            results = self.search(query_vector, k=3)
            
            print(f"\n检索结果 (top-3):")
            for i, result in enumerate(results):
                status = "✅" if not result['text'].startswith("[已删除]") else "❌"
                print(f"  {i+1}. {status} 相似度: {result['similarity']:.4f}")
                print(f"     文本: {result['text'][:60]}...")
                print(f"     事实ID: {result['fact_id']}")
        else:
            print("❌ 没有可用的查询向量")
    
    def demo_add(self):
        """演示添加"""
        print("\n=== 添加演示 ===")
        
        # 创建新事实
        new_fact = "东京是日本首都"
        print(f"添加新事实: {new_fact}")
        
        # 生成随机向量（实际应用中应通过适配器生成）
        dimension = self.vectors.shape[1]
        new_vector = np.random.randn(1, dimension).astype(np.float32)
        faiss.normalize_L2(new_vector)
        
        new_id = self.add_fact(new_fact, "japan_capital", new_vector)
        
        # 立即验证
        print(f"\n立即验证添加结果:")
        results = self.search(new_vector, k=3)
        
        for i, result in enumerate(results):
            if result['index'] == new_id:
                print(f"✅ 成功找到新添加的事实!")
                print(f"   索引: {result['index']}")
                print(f"   相似度: {result['similarity']:.4f}")
                print(f"   文本: {result['text']}")
                break
    
    def demo_delete(self):
        """演示删除"""
        print("\n=== 删除演示 ===")
        
        if len(self.fact_texts) == 0:
            print("❌ 没有可删除的事实")
            return
        
        # 显示可删除的事实
        print("当前事实:")
        for i in range(min(5, len(self.fact_texts))):
            if not self.fact_texts[i].startswith("[已删除]"):
                print(f"  [{i}] {self.fact_texts[i][:50]}...")
        
        try:
            idx = int(input("\n输入要删除的索引: "))
            self.delete_fact(idx)
        except:
            print("❌ 无效输入")
    
    def demo_update(self):
        """演示更新"""
        print("\n=== 更新演示 ===")
        
        if len(self.fact_texts) == 0:
            print("❌ 没有可更新的事实")
            return
        
        # 找一个未删除的事实
        target_idx = None
        for i in range(len(self.fact_texts)):
            if not self.fact_texts[i].startswith("[已删除]"):
                target_idx = i
                break
        
        if target_idx is None:
            print("❌ 所有事实都已删除")
            return
        
        old_text = self.fact_texts[target_idx]
        new_text = f"{old_text}（已更新）"
        
        print(f"更新事实 [{target_idx}]:")
        print(f"  原文本: {old_text}")
        print(f"  新文本: {new_text}")
        
        # 生成新向量（随机）
        dimension = self.vectors.shape[1]
        new_vector = np.random.randn(1, dimension).astype(np.float32)
        faiss.normalize_L2(new_vector)
        
        new_id = self.update_fact(target_idx, new_text, new_vector)
        
        print(f"✅ 更新完成，新索引: {new_id}")
    
    def run_preset_demo(self):
        """运行预设演示"""
        print("\n" + "="*60)
        print("🎬 预设演示开始")
        print("="*60)
        
        time.sleep(1)
        
        # 1. 初始状态
        print("\n1. 📋 初始状态检查")
        self.list_facts(5)
        print(f"   总事实数: {len([t for t in self.fact_texts if not t.startswith('[已删除]')])}")
        
        time.sleep(1)
        
        # 2. 检索演示
        print("\n2. 🔍 检索演示")
        if len(self.vectors) > 0:
            query_vector = self.vectors[0:1]
            results = self.search(query_vector, k=3)
            
            print(f"   查询: {self.fact_texts[0][:50]}...")
            print(f"   找到 {len(results)} 个结果")
            
            for i, result in enumerate(results[:2]):
                print(f"     {i+1}. {result['text'][:50]}... (相似度: {result['similarity']:.4f})")
        
        time.sleep(1)
        
        # 3. 添加演示
        print("\n3. ➕ 添加新事实演示")
        new_facts = [
            ("东京是日本首都", "japan_capital"),
            ("北京是中国首都", "china_capital"),
            ("华盛顿是美国首都", "usa_capital")
        ]
        
        for fact_text, fact_id in new_facts:
            dimension = self.vectors.shape[1]
            new_vector = np.random.randn(1, dimension).astype(np.float32)
            faiss.normalize_L2(new_vector)
            
            self.add_fact(fact_text, fact_id, new_vector)
            time.sleep(0.5)
        
        print(f"   添加后总事实数: {self.index.ntotal}")
        
        time.sleep(1)
        
        # 4. 删除演示
        print("\n4. 🗑️  删除事实演示")
        if len(self.fact_texts) > 2:
            self.delete_fact(1)
            print(f"   删除后事实[1]: {self.fact_texts[1][:50]}...")
        
        time.sleep(1)
        
        # 5. 更新演示
        print("\n5. 🔄 更新事实演示")
        if len(self.fact_texts) > 3:
            old_text = self.fact_texts[2]
            new_text = f"{old_text} - 更新版本"
            
            dimension = self.vectors.shape[1]
            new_vector = np.random.randn(1, dimension).astype(np.float32)
            faiss.normalize_L2(new_vector)
            
            self.update_fact(2, new_text, new_vector)
            print(f"   更新完成")
        
        time.sleep(1)
        
        # 6. 最终状态
        print("\n6. 📊 最终状态")
        self.list_facts(8)
        print(f"   有效事实数: {len([t for t in self.fact_texts if not t.startswith('[已删除]')])}")
        print(f"   向量总数: {self.index.ntotal}")
        
        print("\n" + "="*60)
        print("🎉 预设演示完成！")
        print("="*60)
    
    def save_state(self):
        """保存当前状态"""
        state_dir = Path("/root/千问白盒化实验/editable_state")
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        index_path = state_dir / "editable_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # 保存元数据
        metadata = {
            "fact_ids": self.fact_ids,
            "fact_texts": self.fact_texts,
            "save_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_vectors": self.index.ntotal,
            "active_facts": len([t for t in self.fact_texts if not t.startswith('[已删除]')])
        }
        
        metadata_path = state_dir / "editable_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 状态已保存:")
        print(f"   索引: {index_path}")
        print(f"   元数据: {metadata_path}")

def main():
    print("=== 阶段二：步骤6 - 可编辑性演示 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建可编辑系统
    memory_system = EditableMemorySystem()
    
    # 运行预设演示
    memory_system.run_preset_demo()
    
    # 保存状态
    memory_system.save_state()
    
    # 可选：进入交互模式
    print("\n=== 交互模式 ===")
    print("是否进入交互模式？(y/n)")
    
    choice = input().strip().lower()
    if choice == 'y':
        memory_system.interactive_demo()
    
    print(f"\n=== 完成 ===")
    print(f"结束