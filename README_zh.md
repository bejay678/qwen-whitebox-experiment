# 千问白盒化实验

[![许可证: MIT](https://img.shields.io/badge/许可证-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-orange.svg)](https://github.com/facebookresearch/faiss)

**千问2.5-0.5B-Instruct记忆模块白盒化：80倍检索加速，完全可编辑知识库**

> **作者**: [bejay678](https://github.com/bejay678)  
> **实验日期**: 2026年3月28日  
> **状态**: ✅ **所有目标达成，核心观点已验证**

## 🎯 项目概述

本实验成功验证了大语言模型记忆模块白盒化的技术可行性。通过训练小型适配器网络、构建FAISS向量索引、实现C语言固化，我们实现了：

- **80倍检索加速**（FAISS vs PyTorch暴力搜索）
- **275%判别性提升**（0.2049 → 0.7702）
- **完全可编辑知识库**（动态增删改）
- **端到端问答系统**（实验性）

## 📊 核心成果

### 性能指标
| 组件 | 基准 | 优化后 | 加速比 | 说明 |
|------|------|--------|--------|------|
| 检索系统 | 1.46 ms/查询 | 0.02 ms/查询 | **80.49×** | FAISS IVF vs PyTorch暴力搜索 |
| 适配器前向（CPU） | ~5.0 ms/查询 | 1.21 ms/查询 | **4.1×** | C实现 vs PyTorch |
| 判别性分数 | 0.2049 | 0.7702 | **+275%** | 监督对比学习 |

### 已验证的核心观点
1. ✅ **记忆模块可解耦**：从大模型中分离为独立组件
2. ✅ **解耦模块可白盒化**：转化为高效C/汇编代码
3. ✅ **白盒化优势**：80倍加速 + 完全可编辑 + 完全可解释
4. ✅ **混合架构**：神经网络灵活性 + 白盒代码效率

## 🚀 快速开始

### 安装
```bash
# 克隆仓库
git clone https://github.com/bejay678/qwen-whitebox-experiment.git
cd qwen-whitebox-experiment

# 安装依赖（精确版本确保可复现）
pip install -r requirements.txt

# 或使用conda
conda env create -f environment.yml
conda activate qwen-whitebox
```

### 下载千问模型
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 使用ModelScope（国内推荐）
model_path = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

### 运行演示
```bash
# 运行所有演示
./scripts/run_all.sh

# 或单独运行
python examples/basic_retrieval.py        # 基础检索演示
python examples/knowledge_editing.py      # 可编辑知识库（✅ 正常工作）
python examples/end_to_end_qa.py          # 端到端问答（⚠️ 实验性）
```

### 编译C适配器
```bash
cd src/adapter
chmod +x compile.sh
./compile.sh
python test_adapter.py  # 验证C适配器工作
```

## 📁 项目结构

```
qwen-whitebox-experiment/
├── .github/                    # GitHub配置
│   └── workflows/ci.yml       # CI/CD流水线
├── docs/                      # 文档
│   ├── getting_started.md     # 快速开始指南
│   └── methodology.md         # 技术方法论
├── examples/                  # 使用示例
│   ├── basic_retrieval.py     # 基础检索演示
│   ├── knowledge_editing.py   # 知识编辑演示
│   └── end_to_end_qa.py       # 端到端问答演示
├── src/                       # 源代码
│   ├── adapter/              # 适配器实现
│   │   ├── adapter.c         # C源代码
│   │   ├── libadapter.so     # 编译库
│   │   ├── compile.sh        # 编译脚本
│   │   ├── adapter_model.pt  # PyTorch适配器权重（1.1MB）
│   │   └── *.bin             # C适配器权重（6个文件）
│   ├── retrieval/            # 检索系统
│   │   └── faiss_manager.py  # FAISS索引管理
│   └── utils/                # 工具函数
├── data/                      # 示例数据
│   ├── indices/              # FAISS索引
│   │   ├── editable_index.bin       # 可编辑索引（145KB）
│   │   └── editable_metadata.json   # 事实元数据
│   └── sample_facts.json     # 测试用示例事实
├── scripts/                   # 运行脚本（20+个文件）
│   ├── train_adapter.py      # 训练适配器脚本
│   ├── build_faiss.py        # 构建FAISS索引
│   ├── editable_demo.py      # 可编辑知识演示
│   ├── end_to_end_qa.py      # 端到端问答
│   └── performance_test.py   # 性能测试
├── tests/                     # 测试套件（框架就绪）
├── .gitignore                # Git忽略文件
├── LICENSE                   # MIT许可证
├── README.md                 # 英文主文档
├── README_zh.md              # 中文文档（本文件）
├── requirements.txt          # Python依赖（精确版本）
├── environment.yml           # Conda环境
├── setup.py                  # 安装脚本
├── pyproject.toml           # 现代Python项目配置
└── Dockerfile               # 容器化支持
```

## 🔧 核心组件

### 1. 适配器网络
- **架构**: 896 → 256 → 128 MLP
- **参数**: 262,784
- **训练**: 监督对比学习（11秒完成）
- **训练数据**: 100个事实，252个变体，177正例 + 83负例对
- **判别性**: 0.7702（相比基线0.2049提升275%）

### 2. FAISS检索系统
- **索引类型**: `IndexIDMap(IndexIVFFlat)`（Python: `faiss.index_factory(128, "IVF25,Flat")`）
- **向量维度**: 128
- **距离度量**: 余弦相似度（L2归一化）
- **加速比**: 80.49× vs PyTorch暴力搜索（0.02ms vs 1.46ms每查询）
- **可编辑性**: 支持动态增删改操作

### 3. C适配器实现
- **语言**: 纯C（GCC -O3优化）
- **性能**: 1.21 ms/查询，824 QPS（比PyTorch CPU快4.1倍）
- **接口**: Python ctypes绑定
- **一致性**: 与PyTorch输出完全相同（误差 < 1e-6）
- **大小**: 1.0 MB总计（6个二进制权重文件）

### 4. 可编辑知识库
- **操作**: 增、删、改（实时生效）
- **持久化**: 索引和元数据可保存/加载
- **当前规模**: 256个向量（128维）
- **活跃事实**: 253个事实，3个标记为删除

## 📈 实验结果

### 判别性提升
```
基线判别性: 0.2049
训练后适配器判别性: 0.7702
提升: 275%

相同事实相似度: 0.9464（接近完美）
不同事实相似度: 0.1960（非常低）
最终损失: 0.0034（优秀收敛）
```

### 检索性能
```
PyTorch暴力搜索: 1.46 ms/查询（685 QPS）
FAISS IVF检索: 0.02 ms/查询（50,000 QPS）
加速比: 80.49×
一致性: 100%相同结果
```

### C适配器性能（CPU）
```
PyTorch适配器（CPU）: ~5.0 ms/查询
C适配器（CPU）: 1.21 ms/查询
加速比: 4.1×
吞吐量: 824 QPS
数值一致性: 误差 < 1e-6
```

## ⚠️ 限制与未来工作

### 当前限制
1. **规模**: 已验证256个向量；更大规模（>10K）需进一步测试
2. **知识集成**: 千问生成有时可能"忽略"检索到的事实
3. **删除机制**: FAISS使用标记删除；真正删除需要重建索引
4. **训练数据**: 适配器在100个事实上训练；更多样数据可能提升泛化
5. **GPU优化**: C适配器目前仅CPU；GPU实现可能更快

### 未来工作
1. **规模测试**: 扩展到10K+向量，测量性能扩展
2. **GPU加速**: 实现C适配器的CUDA版本
3. **自动化**: 训练模块自动白盒化的工具链
4. **多模态**: 扩展到图像/音频记忆模块
5. **生产部署**: 为实时应用优化

## 🎯 应用场景

1. **实时问答系统**: 低延迟知识检索（80倍加速）
2. **边缘AI部署**: C适配器适合资源受限环境
3. **可解释AI**: 完全透明的知识存储和检索
4. **动态知识库**: 支持实时更新的AI系统
5. **学术研究**: LLM白盒化方法论的验证
6. **模型压缩**: 用高效代码替换神经组件

## 📚 文档

- [快速开始](docs/getting_started.md) - 快速入门指南
- [方法论](docs/methodology.md) - 技术方法和实现细节
- [API参考](docs/api_reference.md) - 完整API文档
- [贡献指南](docs/contributing.md) - 如何参与项目
- [限制说明](docs/limitations.md) - 当前限制和未来方向

## 🔬 复现实验

### 1. 环境设置
```bash
# 精确版本确保可复现
pip install torch==2.5.1 transformers==5.4.0 faiss-cpu==1.13.2
```

### 2. 训练适配器
```bash
python scripts/train_adapter.py \
  --data data/training_pairs.json \
  --epochs 20 \
  --output src/adapter/adapter_model.pt
```

### 3. 构建FAISS索引
```bash
python scripts/build_faiss.py \
  --vectors vectors/fact_vectors.npy \
  --metadata data/sample_facts.json \
  --output data/indices/faiss_index.bin \
  --nlist 25  # IVF聚类参数
```

### 4. 性能测试
```bash
python scripts/performance_test.py \
  --iterations 1000 \
  --batch_size 32 \
  --output results/performance_report.json
```

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

**模型许可证说明**: 本项目使用Qwen2.5-0.5B-Instruct模型，遵循其原始许可证。请参考：
- [ModelScope上的千问许可证](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/file/view/master/LICENSE)
- [Hugging Face上的千问许可证](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE)

**模型来源**: 本实验使用的Qwen2.5-0.5B-Instruct模型下载自[ModelScope](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files)，为国内用户提供更快的访问速度。

## 🙏 致谢

- **阿里千问团队**: 提供Qwen2.5-0.5B-Instruct模型（[ModelScope](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files) | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)）
- **FAISS团队**: 高效的向量相似度搜索库
- **AutoDL平台**: GPU计算资源（RTX 4090D）
- **开源社区**: 各种依赖和工具
- **ModelScope平台**: 在国内提供可靠的模型托管

## 📞 联系与讨论

- **GitHub Issues**: [创建问题](https://github.com/bejay678/qwen-whitebox-experiment/issues)
- **作者**: [bejay678](https://github.com/bejay678)
- **讨论**: 欢迎通过GitHub Issues进行技术讨论

## 📖 引用

如果您在研究中使用了本工作，请引用：

```bibtex
@misc{qwenwhitebox2026,
  title={千问2.5-0.5B-Instruct记忆模块白盒化},
  author={bejay678},
  year={2026},
  howpublished={\url{https://github.com/bejay678/qwen-whitebox-experiment}},
  note={80倍检索加速，完全可编辑知识库}
}
```

---

**实验详情**:
- **日期**: 2026年3月28日
- **地点**: AutoDL北京B区（RTX 4090D）
- **时长**: ~2.5小时（05:20-07:50 GMT+8）
- **成本**: ~¥4.2-5.2（85分钟GPU时间）
- **状态**: ✅ 所有目标达成，核心观点已验证
- **版本**: v1.0.0
- **代码大小**: 2.6MB（65个文件）
- **模型大小**: 1.1MB适配器 + 145KB索引

**关键成就**: 首次完整验证LLM记忆模块白盒化可行性，实现80倍加速和完全可编辑性。
