import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random

class FactDataset(Dataset):
    """事实数据集"""
    def __init__(self, data_path, tokenizer, model, max_samples=200):
        self.tokenizer = tokenizer
        self.model = model
        
        # 加载数据
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.facts = data.get("facts", [])
        self.variants = data.get("variants", [])
        
        print(f"加载数据: {len(self.facts)}个事实, {len(self.variants)}个变体")
        
        # 提取hidden_state
        print("提取hidden_state...")
        self.hidden_states = []
        self.fact_ids = []
        
        max_variants = min(max_samples, len(self.variants))
        for i, variant in enumerate(self.variants[:max_variants]):
            if i % 20 == 0:
                print(f"  处理变体 {i+1}/{max_variants}")
            
            hidden = self.extract_hidden_state(variant)
            if hidden is not None:
                self.hidden_states.append(hidden)
                self.fact_ids.append(variant.get("fact_id", 0))
        
        self.hidden_states = np.vstack(self.hidden_states)
        print(f"提取完成: {len(self.hidden_states)}个有效hidden_state")
        
        # 构建样本对
        self.build_pairs()
    
    def extract_hidden_state(self, variant):
        """提取hidden_state - 使用平均池化"""
        try:
            if isinstance(variant, dict):
                text = variant.get("text", "")
            else:
                text = str(variant)
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1].mean(dim=1).float().cpu().numpy()
            
            return hidden_state
        except Exception as e:
            return None
    
    def build_pairs(self):
        """构建训练对"""
        print("构建训练对...")
        
        self.positive_pairs = []
        self.negative_pairs = []
        
        # 按事实ID分组
        fact_groups = {}
        for idx, fact_id in enumerate(self.fact_ids):
            if fact_id not in fact_groups:
                fact_groups[fact_id] = []
            fact_groups[fact_id].append(idx)
        
        # 构建正负样本对
        fact_ids_list = list(fact_groups.keys())
        
        for fact_id, indices in fact_groups.items():
            # 正样本
            if len(indices) >= 2:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        self.positive_pairs.append((indices[i], indices[j]))
            
            # 负样本
            other_fact_ids = [fid for fid in fact_ids_list if fid != fact_id]
            if other_fact_ids and indices:
                anchor_idx = random.choice(indices)
                other_fact_id = random.choice(other_fact_ids)
                if other_fact_id in fact_groups and fact_groups[other_fact_id]:
                    negative_idx = random.choice(fact_groups[other_fact_id])
                    self.negative_pairs.append((anchor_idx, negative_idx))
        
        print(f"构建完成: {len(self.positive_pairs)}正对, {len(self.negative_pairs)}负对")
    
    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):
            i, j = self.positive_pairs[idx]
            label = 1.0
        else:
            i, j = self.negative_pairs[idx - len(self.positive_pairs)]
            label = 0.0
        
        return (
            torch.FloatTensor(self.hidden_states[i]),
            torch.FloatTensor(self.hidden_states[j]),
            torch.FloatTensor([label])
        )

class Adapter(nn.Module):
    """适配器网络"""
    def __init__(self, input_dim=896, hidden_dim=256, output_dim=128, dropout=0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 初始化
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

class ContrastiveLoss(nn.Module):
    """对比损失"""
    def __init__(self, margin=0.8):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        cosine_sim = nn.functional.cosine_similarity(output1, output2)
        loss = torch.mean(
            (1 - label) * torch.pow(cosine_sim, 2) +
            label * torch.pow(torch.clamp(self.margin - cosine_sim, min=0.0), 2)
        )
        return loss

def evaluate_adapter(adapter, dataset, device):
    """评估适配器性能"""
    adapter.eval()
    
    hidden_states = torch.FloatTensor(dataset.hidden_states).to(device)
    with torch.no_grad():
        embeddings = adapter(hidden_states)
    
    embeddings_np = embeddings.cpu().numpy()
    sim_matrix = cosine_similarity(embeddings_np)
    
    # 计算判别性
    same_fact_sims = []
    diff_fact_sims = []
    
    for i in range(len(dataset.fact_ids)):
        for j in range(i+1, len(dataset.fact_ids)):
            sim = sim_matrix[i, j]
            if dataset.fact_ids[i] == dataset.fact_ids[j]:
                same_fact_sims.append(sim)
            else:
                diff_fact_sims.append(sim)
    
    if same_fact_sims and diff_fact_sims:
        return {
            "average_same_sim": float(np.mean(same_fact_sims)),
            "average_diff_sim": float(np.mean(diff_fact_sims)),
            "discriminative_score": float(np.mean(same_fact_sims) - np.mean(diff_fact_sims)),
            "same_fact_count": len(same_fact_sims),
            "diff_fact_count": len(diff_fact_sims)
        }
    
    return None

def main():
    print("=== 千问2.5-0.5B 适配器训练 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n=== 加载模型 ===")
    model_path = "/root/千问白盒化实验/models/Qwen2.5-0.5B-Instruct"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device)
        model.eval()
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ tokenizer加载成功")
    except Exception as e:
        print(f"❌ tokenizer加载失败: {e}")
        return
    
    # 加载数据
    print("\n=== 加载数据 ===")
    data_path = "/root/千问白盒化实验/data/facts_dataset.json"
    
    try:
        dataset = FactDataset(data_path, tokenizer, model, max_samples=200)
        print(f"✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    if len(dataset) == 0:
        print("❌ 没有有效的训练数据")
        return
    
    # 评估原始hidden_state
    print("\n=== 评估原始hidden_state ===")
    original_hidden = torch.FloatTensor(dataset.hidden_states)
    original_embeddings_np = original_hidden.numpy()
    original_sim_matrix = cosine_similarity(original_embeddings_np)
    
    same_fact_sims = []
    diff_fact_sims = []
    for i in range(len(dataset.fact_ids)):
        for j in range(i+1, len(dataset.fact_ids)):
            sim = original_sim_matrix[i, j]
            if dataset.fact_ids[i] == dataset.fact_ids[j]:
                same_fact_sims.append(sim)
            else:
                diff_fact_sims.append(sim)
    
    if same_fact_sims and diff_fact_sims:
        original_disc = np.mean(same_fact_sims) - np.mean(diff_fact_sims)
        print(f"原始判别性分数: {original_disc:.4f}")
        print(f"  相同事实平均相似度: {np.mean(same_fact_sims):.4f}")
        print(f"  不同事实平均相似度: {np.mean(diff_fact_sims):.4f}")
    else:
        print("❌ 无法计算原始判别性")
        original_disc = 0.0
    
    # 创建适配器
    print("\n=== 创建适配器 ===")
    adapter = Adapter(
        input_dim=896,
        hidden_dim=256,
        output_dim=128,
        dropout=0.1
    ).to(device)
    
    print(f"适配器参数: {sum(p.numel() for p in adapter.parameters()):,}")
    
    # 训练设置
    print("\n=== 训练设置 ===")
    criterion = ContrastiveLoss(margin=0.8)
    optimizer = optim.Adam(adapter.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    batch_size = 32
    num_epochs = 30
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"训练配置: {num_epochs}轮, 批次大小{batch_size}, 学习率0.001")
    
    # 训练循环
    print("\n=== 开始训练 ===")
    training_log = []
    
    for epoch in range(num_epochs):
        adapter.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (hidden1, hidden2, labels) in enumerate(dataloader):
            hidden1 = hidden1.to(device)
            hidden2 = hidden2.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            
            output1 = adapter(hidden1)
            output2 = adapter(hidden2)
            loss = criterion(output1, output2, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f"  轮次 {epoch+1}, 批次 {batch_idx}: 损失 {loss.item():.4f}")
        
        scheduler.step()
        
        # 每5轮评估一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            eval_result = evaluate_adapter(adapter, dataset, device)
            if eval_result:
                print(f"\n轮次 {epoch+1} 评估:")
                print(f"  损失: {epoch_loss/batch_count:.4f}")
                print(f"  相同事实相似度: {eval_result['average_same_sim']:.4f}")
                print(f"  不同事实相似度: {eval_result['average_diff_sim']:.4f}")
                print(f"  判别性分数: {eval_result['discriminative_score']:.4f}")
                
                training_log.append({
                    "epoch": epoch + 1,
                    "loss": epoch_loss / batch_count,
                    **eval_result
                })
        
        # 早停检查
        if epoch >= 10 and eval_result and eval_result['discriminative_score'] > 0.4:
            print(f"\n🎉 达到目标判别性分数 (>0.4)，提前停止")
            break
    
    # 最终评估
    print("\n=== 最终评估 ===")
    final_result = evaluate_adapter(adapter, dataset, device)
    
    if final_result:
        print(f"训练前判别性: {original_disc:.4f}")
        print(f"训练后判别性: {final_result['discriminative_score']:.4f}")
        print(f"提升: {final_result['discriminative_score'] - original_disc:.4f}")
        
        # 保存结果
        result_data = {
            "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Qwen2.5-0.5B-Instruct",
            "adapter_config": {
                "input_dim": 896,
                "hidden_dim": 256,
                "output_dim": 128,
                "dropout": 0.1
            },
            "training_config": {
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": 0.001,
                "loss_function": "ContrastiveLoss"
            },
            "original_discriminative": float(original_disc),
            "final_results": final_result,
            "training_log": training_log
        }
        
        # 保存模型
        model_dir = Path("/root/千问白盒化实验/models/adapter")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "adapter_model.pt"
        torch.save({
            'model_state_dict': adapter.state_dict(),
            'config': adapter.network
        }, model_path)
        
        # 保存结果
        result_path = model_dir / "training_results.json"
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 适配器模型已保存: {model_path}")
        print(f"✅ 训练结果已保存: {result_path}")
        
        # 决策建议
        final_disc = final_result['discriminative_score']
        print(f"\n=== 决策建议 ===")
        if final_disc > 0.4:
            print("🎉 优秀！适配器训练成功，判别性 > 0.4")
            print("   建议：立即开始阶段二实验（FAISS固化）")
        elif final_disc > 0.2:
            print("✅ 良好！适配器有效，判别性 > 0.2")
            print("   建议：可以继续实验，考虑优化适配器")
        elif final_disc > 0.1:
            print("⚠️ 一般！适配器有改善但不够强")
            print("   建议：调整适配器架构或训练策略")
        else:
            print("❌ 较差！适配器训练效果不明显")
            print("   建议：重新设计适配器或使用更强损失函数")
    else:
        print("❌ 最终评估失败")
    
    print(f"\n训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()