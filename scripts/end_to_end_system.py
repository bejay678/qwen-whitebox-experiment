import numpy as np
import faiss
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 导入C适配器包装器
sys.path.insert(0, '/root/千问白盒化实验/scripts')
from c_adapter_wrapper import CAdapter

class EndToEndQASystem:
    """端到端问答系统"""
    
    def __init__(self, use_c_adapter=True):
        print("=== 初始化端到端问答系统 ===")
        print(f"使用C适配器: {use_c_adapter}")
        
        self.use_c_adapter = use_c_adapter
        
        # 1. 加载千问模型
        print("\n1. 加载千问模型...")
        self.load_qwen_model()
        
        # 2. 加载适配器
        print("\n2. 加载适配器...")
        self.load_adapter()
        
        # 3. 加载FAISS索引
        print("\n3. 加载FAISS索引...")
        self.load_faiss_index()
        
        # 4. 加载事实元数据
        print("\n4. 加载事实元数据...")
        self.load_fact_metadata()
        
        print(f"\n✅ 系统初始化完成")
        print(f"   模型: Qwen2.5-0.5B-Instruct")
        print(f"   适配器: {'C语言实现' if use_c_adapter else 'PyTorch实现'}")
        print(f"   索引向量: {self.index.ntotal} 个")
        print(f"   事实数量: {len(self.fact_texts)} 个")
    
    def load_qwen_model(self):
        """加载千问模型"""
        model_path = "/root/千问白盒化实验/models/Qwen2.5-0.5B-Instruct"
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            self.model.eval()
            print(f"✅ 千问模型加载成功")
        except Exception as e:
            print(f"❌ 千问模型加载失败: {e}")
            raise
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f"✅ tokenizer加载成功")
        except Exception as e:
            print(f"❌ tokenizer加载失败: {e}")
            raise
    
    def load_adapter(self):
        """加载适配器"""
        if self.use_c_adapter:
            # 使用C适配器
            try:
                self.adapter = CAdapter()
                self.adapter_type = "c"
                print(f"✅ C适配器加载成功")
            except Exception as e:
                print(f"❌ C适配器加载失败: {e}")
                # 回退到PyTorch适配器
                self.load_pytorch_adapter()
        else:
            # 使用PyTorch适配器
            self.load_pytorch_adapter()
    
    def load_pytorch_adapter(self):
        """加载PyTorch适配器"""
        print("加载PyTorch适配器...")
        adapter_path = "/root/千问白盒化实验/models/adapter/adapter_model.pt"
        
        try:
            checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
        except:
            checkpoint = torch.load(adapter_path, map_location="cpu")
        
        state_dict = checkpoint['model_state_dict']
        
        # 重建PyTorch适配器
        from torch import nn
        
        class PyTorchAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(896, 256)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
                self.fc2 = nn.Linear(256, 128)
                self.ln = nn.LayerNorm(128)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.ln(x)
                return x
        
        self.adapter = PyTorchAdapter()
        
        # 修复键名
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key == "network.0.weight":
                fixed_state_dict["fc1.weight"] = value
            elif key == "network.0.bias":
                fixed_state_dict["fc1.bias"] = value
            elif key == "network.3.weight":
                fixed_state_dict["fc2.weight"] = value
            elif key == "network.3.bias":
                fixed_state_dict["fc2.bias"] = value
            elif key == "network.4.weight":
                fixed_state_dict["ln.weight"] = value
            elif key == "network.4.bias":
                fixed_state_dict["ln.bias"] = value
        
        self.adapter.load_state_dict(fixed_state_dict)
        self.adapter.eval()
        self.adapter_type = "pytorch"
        print(f"✅ PyTorch适配器加载成功")
    
    def load_faiss_index(self):
        """加载FAISS索引"""
        index_path = "/root/千问白盒化实验/editable_state/editable_index.bin"
        
        try:
            self.index = faiss.read_index(str(index_path))
            print(f"✅ FAISS索引加载成功: {self.index.ntotal} 个向量")
        except Exception as e:
            print(f"❌ FAISS索引加载失败: {e}")
            raise
    
    def load_fact_metadata(self):
        """加载事实元数据"""
        metadata_path = "/root/千问白盒化实验/editable_state/editable_metadata.json"
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.fact_texts = metadata.get("fact_texts", [])
            print(f"✅ 事实元数据加载成功: {len(self.fact_texts)} 个事实")
        except Exception as e:
            print(f"❌ 事实元数据加载失败: {e}")
            self.fact_texts = []
    
    def extract_hidden_state(self, text):
        """提取hidden_state（平均池化）"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1].mean(dim=1)  # 平均池化
            
            return hidden_state.squeeze(0).numpy().astype(np.float32)
        except Exception as e:
            print(f"❌ hidden_state提取失败: {e}")
            return np.zeros(896, dtype=np.float32)
    
    def get_query_vector(self, text):
        """获取查询向量（通过适配器）"""
        # 1. 提取hidden_state
        hidden_state = self.extract_hidden_state(text)
        
        # 2. 通过适配器
        if self.adapter_type == "c":
            # 使用C适配器
            query_vector = self.adapter.forward(hidden_state)
        else:
            # 使用PyTorch适配器
            with torch.no_grad():
                hidden_tensor = torch.from_numpy(hidden_state).unsqueeze(0)
                output_tensor = self.adapter(hidden_tensor)
                query_vector = output_tensor.squeeze(0).numpy()
        
        # 3. 归一化（用于余弦相似度）
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        return query_vector
    
    def retrieve_facts(self, query_text, k=5):
        """检索相关事实"""
        print(f"\n🔍 检索: '{query_text[:50]}...'")
        
        start_time = time.perf_counter()
        
        # 获取查询向量
        query_vector = self.get_query_vector(query_text)
        
        # FAISS检索
        distances, indices = self.index.search(query_vector, k)
        
        retrieval_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # 构建结果
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.fact_texts):
                fact_text = self.fact_texts[idx]
                # 跳过已删除的事实
                if not fact_text.startswith("[已删除]"):
                    results.append({
                        "index": int(idx),
                        "similarity": float(distances[0][i]),
                        "text": fact_text
                    })
        
        print(f"   检索耗时: {retrieval_time:.2f} ms")
        print(f"   找到 {len(results)} 个相关事实")
        
        return results, retrieval_time
    
    def build_prompt(self, query, facts, max_facts=3):
        """构建prompt"""
        # 选择top事实
        selected_facts = facts[:max_facts]
        
        # 构建prompt
        prompt = "已知事实：\n"
        
        for i, fact in enumerate(selected_facts):
            prompt += f"{i+1}. {fact['text']}\n"
        
        prompt += f"\n问题：{query}\n回答："
        
        return prompt
    
    def generate_answer(self, prompt, max_length=100):
        """生成答案"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return answer.strip()
        except Exception as e:
            print(f"❌ 答案生成失败: {e}")
            return "抱歉，生成答案时出现错误。"
    
    def ask_question(self, question, k=5, max_facts=3):
        """完整问答流程"""
        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"{'='*60}")
        
        total_start = time.perf_counter()
        
        # 1. 检索相关事实
        facts, retrieval_time = self.retrieve_facts(question, k)
        
        # 2. 构建prompt
        if facts:
            prompt = self.build_prompt(question, facts, max_facts)
            print(f"\n📝 构建的prompt:")
            print(f"{'-'*40}")
            print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
            print(f"{'-'*40}")
        else:
            print("⚠️  未找到相关事实，直接回答问题")
            prompt = f"问题：{question}\n回答："
        
        # 3. 生成答案
        generation_start = time.perf_counter()
        answer = self.generate_answer(prompt)
        generation_time = (time.perf_counter() - generation_start) * 1000
        
        # 4. 输出结果
        total_time = (time.perf_counter() - total_start) * 1000
        
        print(f"\n💡 答案: {answer}")
        print(f"\n⏱️  性能指标:")
        print(f"   检索耗时: {retrieval_time:.2f} ms")
        print(f"   生成耗时: {generation_time:.2f} ms")
        print(f"   总耗时: {total_time:.2f} ms")
        
        if facts:
            print(f"\n📚 参考事实:")
            for i, fact in enumerate(facts[:3]):
                print(f"   {i+1}. {fact['text'][:80]}... (相似度: {fact['similarity']:.4f})")
        
        return {
            "question": question,
            "answer": answer,
            "facts": facts,
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": generation_time,
            "total_time_ms": total_time,
            "adapter_type": self.adapter_type
        }
    
    def performance_test(self, test_questions=None):
        """性能测试"""
        if test_questions is None:
            test_questions = [
                "巴黎是哪个国家的首都？",
                "柏林是哪个国家的首都？",
                "东京是哪个国家的首都？",
                "北京是哪个国家的首都？",
                "华盛顿是哪个国家的首都？"
            ]
        
        print(f"\n{'='*60}")
        print(f"📊 性能测试 ({self.adapter_type.upper()}适配器)")
        print(f"{'='*60}")
        
        results = []
        
        for question in test_questions:
            result = self.ask_question(question, k=3, max_facts=2)
            results.append(result)
            time.sleep(0.5)  # 避免过热
        
        # 统计结果
        retrieval_times = [r["retrieval_time_ms"] for r in results]
        generation_times = [r["generation_time_ms"] for r in results]
        total_times = [r["total_time_ms"] for r in results]
        
        print(f"\n{'='*60}")
        print(f"📈 性能统计")
        print(f"{'='*60}")
        print(f"测试问题数: {len(results)}")
        print(f"平均检索耗时: {np.mean(retrieval_times):.2f} ± {np.std(retrieval_times):.2f} ms")
        print(f"平均生成耗时: {np.mean(generation_times):.2f} ± {np.std(generation_times):.2f} ms")
        print(f"平均总耗时: {np.mean(total_times):.2f} ± {np.std(total_times):.2f} ms")
        print(f"适配器类型: {self.adapter_type}")
        
        return results

def main():
    """主函数"""
    print("=== 端到端问答系统演示 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建系统（使用C适配器）
    print("\n创建系统（使用C适配器）...")
    system_c = EndToEndQASystem(use_c_adapter=True)
    
    # 演示问答
    print("\n" + "="*60)
    print("🎭 演示问答")
    print("="*60)
    
    demo_questions = [
        "巴黎是哪个国家的首都？",
        "柏林是哪个国家的首都？",
        "东京是哪个国家的首都？"
    ]
    
    for question in demo_questions:
        system_c.ask_question(question)
        print("\n" + "-"*60)
    
    # 性能测试
    print("\n" + "="*60)
    print("📊 性能对比测试")
    print("="*60)
    
    # C适配器性能
    print("\n1. C适配器性能测试")
    results_c = system_c.performance_test()
    
    # PyTorch适配器性能
    print("\n2. PyTorch适配器性能测试")
    system_pytorch = EndToEndQASystem(use_c_adapter=False)
    results_pytorch = system_pytorch.performance_test()
    
    # 对比结果
    print("\n" + "="*60)
    print("📈 性能对比")
    print("="*60)
    
    avg_c = np.mean([r["retrieval_time_ms"] for r in results_c])
    avg_pytorch = np.mean([r["retrieval_time_ms"] for r in results_pytorch])
    
    if avg_pytorch > 0:
        speedup = avg_pytorch / avg_c
        print(f"检索加速比: {speedup:.2f}倍")
        print(f"  C适配器: {avg_c:.2f} ms")
        print(f"  PyTorch适配器: {avg_pytorch:.2f} ms")
    else:
        print("无法计算加速比")
    
    # 保存结果
    print(f"\n💾 保存结果...")
    results_dir = Path("/root/千问白盒化实验/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "test_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "c_adapter_results": results_c,
        "pytorch_adapter_results": results_pytorch,
        "performance_comparison": {
            "c_adapter_avg_retrieval_ms": float(avg_c),
            "pytorch_adapter_avg_retrieval_ms": float(avg_pytorch),
            "retrieval_speedup": float(speedup) if avg_pytorch > 0 else 0
        }
    }
    
    result_path = results_dir / "end_to_end_results.json"
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存: {result_path}")
    
    print(f"\n=== 演示完成 ===")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()