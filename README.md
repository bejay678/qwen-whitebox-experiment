# Qwen White-box Experiment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-orange.svg)](https://github.com/facebookresearch/faiss)

**White-boxing memory modules of Qwen2.5-0.5B-Instruct: 80× retrieval acceleration, fully editable knowledge base**

> **Author**: [bejay678](https://github.com/bejay678)  
> **Experiment Date**: March 28, 2026  
> **Status**: ✅ All objectives achieved, core insights validated

## 🎯 Overview

This experiment successfully validates the technical feasibility of white-boxing memory modules in large language models. By training a small adapter network, building FAISS vector indices, and implementing C-language solidification, we achieved:

- **80× retrieval acceleration** (FAISS vs PyTorch brute-force)
- **275% discriminative power improvement** (0.2049 → 0.7702)
- **Fully editable knowledge base** (dynamic add/delete/update)
- **End-to-end question answering system** (experimental)

## 📊 Key Results

### Performance Metrics
| Component | Baseline | Optimized | Speedup | Note |
|-----------|----------|-----------|---------|------|
| Retrieval System | 1.46 ms/query | 0.02 ms/query | **80.49×** | FAISS IVF vs PyTorch brute-force |
| Adapter Forward (CPU) | ~5.0 ms/query | 1.21 ms/query | **4.1×** | C implementation vs PyTorch |
| Discriminative Score | 0.2049 | 0.7702 | **+275%** | Supervised contrastive learning |

### Core Validated Insights
1. ✅ **Memory modules can be decoupled** from LLMs as independent components
2. ✅ **Decoupled modules can be transformed** into efficient white-box code (C/assembly)
3. ✅ **White-boxing provides**: 80× acceleration + full editability + complete interpretability
4. ✅ **Hybrid architecture combines** neural network flexibility with white-box code efficiency

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/bejay678/qwen-whitebox-experiment.git
cd qwen-whitebox-experiment

# Install dependencies (exact versions for reproducibility)
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate qwen-whitebox
```

### Download Qwen Model

**Source**: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files) (recommended for users in China)  
**Alternative**: [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Using ModelScope (recommended for China)
model_path = "Qwen/Qwen2.5-0.5B-Instruct"
# or explicitly: model_path = "modelscope/Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

### Run Demos
```bash
# Run all demonstrations
./scripts/run_all.sh

# Or run individually
python examples/basic_retrieval.py        # Basic FAISS retrieval
python examples/knowledge_editing.py      # Editable knowledge base (✅ Working)
python examples/end_to_end_qa.py          # Complete Q&A system (⚠️ Experimental)
```

### Compile C Adapter
```bash
cd src/adapter
chmod +x compile.sh
./compile.sh
python test_adapter.py  # Verify C adapter works
```

## 📁 Project Structure

```
qwen-whitebox-experiment/
├── .github/                    # GitHub configurations
│   └── workflows/ci.yml       # CI/CD pipeline
├── docs/                      # Documentation
│   ├── getting_started.md     # Quick start guide
│   └── methodology.md         # Technical methodology
├── examples/                  # Usage examples
│   ├── basic_retrieval.py     # Basic retrieval demo
│   ├── knowledge_editing.py   # Knowledge editing demo
│   └── end_to_end_qa.py       # End-to-end Q&A demo
├── src/                       # Source code
│   ├── adapter/              # Adapter implementation
│   │   ├── adapter.c         # C source code
│   │   ├── libadapter.so     # Compiled library
│   │   ├── compile.sh        # Compilation script
│   │   ├── adapter_model.pt  # PyTorch adapter weights (1.1MB)
│   │   └── *.bin             # C adapter weights (6 files)
│   ├── retrieval/            # Retrieval system
│   │   └── faiss_manager.py  # FAISS index management
│   └── utils/                # Utility functions
├── data/                      # Example data
│   ├── indices/              # FAISS indices
│   │   ├── editable_index.bin       # Editable index (145KB)
│   │   └── editable_metadata.json   # Fact metadata
│   └── sample_facts.json     # Sample facts for testing
├── scripts/                   # Run scripts (20+ files)
│   ├── train_adapter.py      # Train adapter script
│   ├── build_faiss.py        # Build FAISS index
│   ├── editable_demo.py      # Editable knowledge demo
│   ├── end_to_end_qa.py      # End-to-end Q&A
│   └── performance_test.py   # Performance testing
├── tests/                     # Test suite (framework ready)
├── .gitignore                # Git ignore file
├── LICENSE                   # MIT License
├── README.md                 # This file
├── requirements.txt          # Python dependencies (exact versions)
├── environment.yml           # Conda environment
├── setup.py                  # Installation script
├── pyproject.toml           # Modern Python project config
└── Dockerfile               # Containerization support
```

## 🔧 Core Components

### 1. Adapter Network
- **Architecture**: 896 → 256 → 128 MLP
- **Parameters**: 262,784
- **Training**: Supervised contrastive learning (11 seconds)
- **Training Data**: 100 facts, 252 variations, 177 positive + 83 negative pairs
- **Discriminative Power**: 0.7702 (+275% improvement from 0.2049)

### 2. FAISS Retrieval System
- **Index Type**: `IndexIDMap(IndexIVFFlat)` (Python: `faiss.index_factory(128, "IVF25,Flat")`)
- **Vector Dimension**: 128
- **Distance Metric**: Cosine similarity (L2 normalized)
- **Speedup**: 80.49× vs PyTorch brute-force (0.02ms vs 1.46ms per query)
- **Editable**: Supports dynamic add/delete/update operations

### 3. C Adapter Implementation
- **Language**: Pure C (GCC with -O3 optimization)
- **Performance**: 1.21 ms/query, 824 QPS (4.1× faster than PyTorch on CPU)
- **Interface**: Python ctypes binding
- **Consistency**: Identical output to PyTorch (error < 1e-6)
- **Size**: 1.0 MB total (6 binary weight files)

### 4. Editable Knowledge Base
- **Operations**: Add, delete, update (real-time effect)
- **Persistence**: Index and metadata can be saved/loaded
- **Current Scale**: 256 vectors (128-dimensional)
- **Active Facts**: 253 facts, 3 marked as deleted

## 📈 Experimental Results

### Discriminative Power Improvement
```
Baseline discriminative power: 0.2049
Trained adapter discriminative power: 0.7702
Improvement: 275%

Same-fact similarity: 0.9464 (near perfect)
Different-fact similarity: 0.1960 (very low)
Final loss: 0.0034 (excellent convergence)
```

### Retrieval Performance
```
PyTorch brute-force retrieval: 1.46 ms/query (685 QPS)
FAISS IVF retrieval: 0.02 ms/query (50,000 QPS)
Speedup: 80.49×
Consistency: 100% identical results
```

### C Adapter Performance (CPU)
```
PyTorch adapter (CPU): ~5.0 ms/query
C adapter (CPU): 1.21 ms/query
Speedup: 4.1×
Throughput: 824 QPS
Numerical consistency: Error < 1e-6
```

## ⚠️ Limitations & Future Work

### Current Limitations
1. **Scale**: Validated with 256 vectors; larger scales (>10K) need further testing
2. **Knowledge Integration**: Qwen generation may "ignore" retrieved facts in some cases
3. **Deletion Mechanism**: FAISS uses marking deletion; true deletion requires index rebuild
4. **Training Data**: Adapter trained on 100 facts; more diverse data may improve generalization
5. **GPU Optimization**: C adapter currently CPU-only; GPU implementation could be faster

### Future Work
1. **Scale Testing**: Extend to 10K+ vectors and measure performance scaling
2. **GPU Acceleration**: Implement CUDA version of C adapter
3. **Automation**: Toolchain for automatic white-boxing of trained modules
4. **Multi-modal**: Extend to image/audio memory modules
5. **Production Deployment**: Optimize for real-time applications

## 🎯 Applications

1. **Real-time Q&A Systems**: Low-latency knowledge retrieval (80× faster)
2. **Edge AI Deployment**: C adapter suitable for resource-constrained environments
3. **Explainable AI**: Fully transparent knowledge storage and retrieval
4. **Dynamic Knowledge Bases**: AI systems supporting real-time updates
5. **Academic Research**: Validation of LLM white-boxing methodology
6. **Model Compression**: Replace neural components with efficient code

## 📚 Documentation

- [Getting Started](docs/getting_started.md) - Quick start guide
- [Methodology](docs/methodology.md) - Technical approach and implementation details
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Contributing](docs/contributing.md) - How to contribute to the project
- [中文文档](README_zh.md) - Chinese documentation

## 🔬 Reproducing the Experiment

### 1. Environment Setup
```bash
# Exact versions for reproducibility
pip install torch==2.5.1 transformers==5.4.0 faiss-cpu==1.13.2
```

### 2. Train the Adapter
```bash
python scripts/train_adapter.py \
  --data data/training_pairs.json \
  --epochs 20 \
  --output src/adapter/adapter_model.pt
```

### 3. Build FAISS Index
```bash
python scripts/build_faiss.py \
  --vectors vectors/fact_vectors.npy \
  --metadata data/sample_facts.json \
  --output data/indices/faiss_index.bin \
  --nlist 25  # IVF clustering parameter
```

### 4. Performance Testing
```bash
python scripts/performance_test.py \
  --iterations 1000 \
  --batch_size 32 \
  --output results/performance_report.json
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Model License Note**: This project uses Qwen2.5-0.5B-Instruct under its original license. Please refer to:
- [Qwen LICENSE on ModelScope](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/file/view/master/LICENSE)
- [Qwen LICENSE on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE)

**Model Source**: The Qwen2.5-0.5B-Instruct model used in this experiment was downloaded from [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files), which provides faster access for users in China.

## 🙏 Acknowledgments

- **Alibaba Qwen Team**: For the Qwen2.5-0.5B-Instruct model ([ModelScope](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files) | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct))
- **FAISS Team**: For the efficient vector similarity search library
- **AutoDL Platform**: For GPU computing resources (RTX 4090D)
- **Open Source Community**: For various dependencies and tools
- **ModelScope Platform**: For providing reliable model hosting in China

## 📞 Contact & Discussion

- **GitHub Issues**: [Create an issue](https://github.com/bejay678/qwen-whitebox-experiment/issues)
- **Author**: [bejay678](https://github.com/bejay678)
- **Discussion**: Technical discussions welcome via GitHub Issues

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{qwenwhitebox2026,
  title={White-boxing Memory Modules in Qwen2.5-0.5B-Instruct},
  author={bejay678},
  year={2026},
  howpublished={\url{https://github.com/bejay678/qwen-whitebox-experiment}},
  note={80× retrieval acceleration, fully editable knowledge base}
}
```

---

**Experiment Details**:
- **Date**: March 28, 2026
- **Location**: AutoDL Beijing Zone B (RTX 4090D)
- **Duration**: ~2.5 hours (05:20-07:50 GMT+8)
- **Cost**: ~¥4.2-5.2 (85 minutes GPU time)
- **Status**: ✅ All objectives achieved, core insights validated
- **Version**: v1.0.0
- **Code Size**: 2.6MB (62 files)
- **Model Size**: 1.1MB adapter + 145KB index

**Key Achievement**: First complete validation of LLM memory module white-boxing feasibility with 80× acceleration and full editability.
