import sys
import numpy as np
import faiss
import json
import time
from pathlib import Path
import torch

# 导入C适配器包装器
sys.path.insert(0, '/root/千问白盒化实验/scripts')

print("=== 端到端问答系统演示 ===")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 先测试C适配器包装器
print("\n测试C适配器包装器...")
try:
    from c_adapter_wrapper import CAdapter
    adapter = CAdapter()
    print("✅ C适配器包装器测试通过")
    
    # 简单测试
    test_input = np.random.randn(896).astype(np.float32)
    output = adapter.forward(test_input)
    print(f"测试输入: {test_input.shape} → 输出: {output.shape}")
    
except Exception as e:
    print(f"❌ C适配器包装器测试失败: {e}")
    print("将使用PyTorch适配器")

# 现在加载其他模块
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleQASystem:
    """简化版问答系统"""
    
    def __init__(self, use_c_adapter=True):
        print(f"\n初始化问答系统 (使用{'C' if use_c_adapter else 'PyTorch'}适配器)...")
        self.use_c_adapter = use_c_adapter
        
        # 加载模型
        print("加载千问模型...")
        model_path = "/root/千问白盒化实验/models/Qwen2.5-0.5B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载适配器
        if use_c_adapter:
            try:
                self.adapter = CAdapter()
                self.adapter_type = "c"
            except:
                print("C适配器失败，回退到PyTorch")
                self.load_pytorch_adapter()
        else:
            self.load_pytorch_adapter()
        
        # 加载FAISS索引
        print("加载FAISS索引...")
        index_path = "/root/千问白盒化实验/editable_state/editable_index.bin"
        self.index = faiss.read_index(str(index_path))
        
        # 加载元数据
        metadata_path = "/root/千问白盒化实验/editable_state/editable_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.fact_texts = metadata.get("fact_texts", [])
        
        print(f"✅ 系统初始化完成")
        print(f"   向量: {self.index.ntotal}个, 事实: {len(self.fact_texts)}个")
    
    def load_pytorch_adapter(self):
        """加载PyTorch适配器"""
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
        self.adapter.eval()
        self.adapter_type = "pytorch"
    
    def ask_question(self, question):
        """问答"""
        print(f"\n❓ 问题: {question}")
        
        start_time = time.perf_counter()
        
        # 提取hidden_state
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1].mean(dim=1).squeeze(0).numpy().astype(np.float32)
        
        # 通过适配器
        if self.adapter_type == "c":
            query_vector = self.adapter.forward(hidden_state)
        else:
            with torch.no_grad():
                hidden_tensor = torch.from_numpy(hidden_state).unsqueeze(0)
                output_tensor = self.adapter(hidden_tensor)
                query_vector = output_tensor.squeeze(0).numpy()
        
        # 检索
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, 3)
        
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        # 构建prompt
        prompt = "已知事实：\n"
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.fact_texts):
                fact = self.fact_texts[idx]
                if not fact.startswith("[已删除]"):
                    prompt += f"{i+1}. {fact}\n"
        
        prompt += f"\n问题：{question}\n回答："
        
        # 生成答案
        gen_start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
        
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generation_time = (time.perf_counter() - gen_start) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        print(f"💡 答案: {answer}")
        print(f"⏱️  检索: {retrieval_time:.1f}ms, 生成: {generation_time:.1f}ms, 总计: {total_time:.1f}ms")
        
        return {
            "question": question,
            "answer": answer,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "adapter_type": self.adapter_type
        }

def main():
    """主演示"""
    print("\n" + "="*60)
    print("🎭 端到端问答演示")
    print("="*60)
    
    # 使用C适配器
    print("\n1. 使用C适配器")
    system_c = SimpleQASystem(use_c_adapter=True)
    
    questions = [
        "巴黎是哪个国家的首都？",
        "柏林是哪个国家的首都？",
        "东京是哪个国家的首都？"
    ]
    
    c_results = []
    for q in questions:
        result = system_c.ask_question(q)
        c_results.append(result)
        time.sleep(0.5)
    
    # 使用PyTorch适配器
    print("\n2. 使用PyTorch适配器")
    system_pytorch = SimpleQASystem(use_c_adapter=False)
    
    pytorch_results = []
    for q in questions:
        result = system_pytorch.ask_question(q)
        pytorch_results.append(result)
        time.sleep(0.5)
    
    # 对比
    print("\n" + "="*60)
    print("📊 性能对比")
    print("="*60)
    
    c_avg = np.mean([r["retrieval_time"] for r in c_results])
    pytorch_avg = np.mean([r["retrieval_time"] for r in pytorch_results])
    
    if pytorch_avg > 0:
        speedup = pytorch_avg / c_avg
        print(f"检索加速比: {speedup:.2f}倍")
        print(f"  C适配器平均: {c_avg:.1f}ms")
        print(f"  PyTorch适配器平均: {pytorch_avg:.1f}ms")
    
    # 保存结果
    result_data = {
        "test_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "c_results": c_results,
        "pytorch_results": pytorch_results,
        "performance": {
            "c_avg_retrieval": float(c_avg),
            "pytorch_avg_retrieval": float(pytorch_avg),
            "speedup": float(speedup) if pytorch_avg > 0 else 0
        }
    }
    
    result_path = "/root/千问白盒化实验/results/end_to_end_demo.json"
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {result_path}")
    print(f"\n=== 演示完成 ===")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()