# Getting Started

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows (WSL2 recommended for Windows)

### For GPU Acceleration
- **NVIDIA GPU** with CUDA 11.8 or higher
- **CUDA Toolkit** installed
- **cuDNN** libraries

## Installation

### Option 1: Using pip (Recommended)
```bash
# Clone the repository
git clone https://github.com/username/qwen-whitebox-experiment.git
cd qwen-whitebox-experiment

# Install with pip
pip install -e .

# Or install with GPU support
pip install -e .[gpu]

# For development
pip install -e .[dev]
```

### Option 2: Using conda
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate qwen-whitebox

# Install in development mode
pip install -e .
```

### Option 3: Manual Installation
```bash
# Install core dependencies
pip install torch transformers faiss-cpu numpy scikit-learn tqdm

# Clone repository
git clone https://github.com/username/qwen-whitebox-experiment.git
cd qwen-whitebox-experiment

# Add to Python path
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Quick Verification

### Test Installation
```python
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import faiss
print(f"FAISS: {faiss.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")
```

### Run Basic Tests
```bash
# Test adapter
python -m pytest tests/test_adapter.py -v

# Test retrieval
python -m pytest tests/test_retrieval.py -v

# Run all tests
python -m pytest tests/ -v
```

## First Steps

### 1. Explore the Project Structure
```bash
# View project structure
tree -I "__pycache__|*.pyc" -L 3

# Or use find
find . -type f -name "*.py" | head -20
```

### 2. Download Qwen Model

**Recommended Source (China)**: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files)  
**Alternative Source**: [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Using ModelScope (faster in China)
model_path = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Note: First download may take time (~954MB)
# The model will be cached in ~/.cache/huggingface/hub/
```

### 3. Run the Basic Demo
```bash
# Run the complete demonstration pipeline
./scripts/run_all.sh

# Or run individual components
python examples/basic_retrieval.py
python examples/knowledge_editing.py
python examples/end_to_end_qa.py
```

### 3. Compile C Adapter
```bash
cd src/adapter/c_implementation
chmod +x compile.sh
./compile.sh

# Test C adapter
python test_adapter.py
```

## Basic Usage Examples

### Example 1: Basic Retrieval
```python
from src.retrieval.faiss_manager import FAISSManager

# Initialize FAISS manager
manager = FAISSManager("data/indices/editable_index.bin")

# Load metadata
metadata = manager.load_metadata("data/indices/editable_metadata.json")

# Generate a query vector (in practice, from adapter)
import numpy as np
query_vector = np.random.randn(1, 128).astype(np.float32)

# Search
results = manager.search(query_vector, k=5)
print(f"Found {len(results)} results")
```

### Example 2: Knowledge Editing
```python
from src.retrieval.vector_store import EditableVectorStore

# Create editable store
store = EditableVectorStore(dimension=128)

# Add facts
store.add_fact("Paris is the capital of France", vector1)
store.add_fact("Berlin is the capital of Germany", vector2)

# Delete a fact
store.delete_fact(fact_id=0)

# Update a fact
store.update_fact(fact_id=1, new_text="Berlin is Germany's capital", new_vector=vector2_updated)

# Save to disk
store.save("my_knowledge_base")
```

### Example 3: End-to-End Q&A
```python
from src.utils.model_loader import load_qwen_model
from src.adapter.pytorch_adapter import PyTorchAdapter
from src.retrieval.faiss_manager import FAISSManager

# Load components
model, tokenizer = load_qwen_model()
adapter = PyTorchAdapter.load("models/adapter_model.pt")
faiss_manager = FAISSManager("data/indices/editable_index.bin")

# Ask a question
question = "What is the capital of France?"

# Extract hidden state
inputs = tokenizer(question, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_state = outputs.hidden_states[-1].mean(dim=1)

# Get query vector
query_vector = adapter(hidden_state).numpy()

# Retrieve facts
facts = faiss_manager.search(query_vector, k=3)

# Build prompt and generate answer
prompt = build_prompt(question, facts)
answer = generate_answer(model, tokenizer, prompt)
print(f"Answer: {answer}")
```

## Configuration

### Environment Variables
```bash
# Set these in your shell or .env file
export QWEN_MODEL_PATH="./models/Qwen2.5-0.5B-Instruct"
export FAISS_INDEX_PATH="./data/indices/editable_index.bin"
export ADAPTER_MODEL_PATH="./models/adapter_model.pt"
export CUDA_VISIBLE_DEVICES="0"  # Use GPU 0
```

### Configuration File
Create `config.yaml`:
```yaml
model:
  path: "./models/Qwen2.5-0.5B-Instruct"
  dtype: "float32"
  
adapter:
  input_dim: 896
  hidden_dim: 256
  output_dim: 128
  model_path: "./models/adapter_model.pt"
  
retrieval:
  index_path: "./data/indices/editable_index.bin"
  metadata_path: "./data/indices/editable_metadata.json"
  search_k: 5
  nprobe: 10
  
performance:
  batch_size: 32
  use_c_adapter: true
  use_gpu: true
```

## Troubleshooting

### Common Issues

#### Issue 1: CUDA not available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU version if no GPU
pip install faiss-cpu torch --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 2: FAISS installation fails
```bash
# Try different installation methods
pip install faiss-cpu --no-cache-dir

# Or build from source
conda install -c conda-forge faiss-cpu
```

#### Issue 3: Memory issues
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Use float16
import torch
torch.set_default_dtype(torch.float16)

# Clear cache
torch.cuda.empty_cache()
```

#### Issue 4: C adapter compilation fails
```bash
# Check compiler
gcc --version

# Install build tools
sudo apt-get install build-essential  # Ubuntu/Debian
brew install gcc                      # macOS

# Manual compilation
cd src/adapter/c_implementation
gcc -O3 -fPIC -shared adapter.c -o libadapter.so -lm
```

## Next Steps

### Learn More
1. Read the [Methodology](methodology.md) document
2. Explore the [API Reference](api_reference.md)
3. Check out the [examples](../examples/) directory

### Extend the Project
1. Modify adapter architecture in `src/adapter/`
2. Add new retrieval methods in `src/retrieval/`
3. Integrate with other LLMs

### Contribute
1. Read [Contributing Guidelines](contributing.md)
2. Fork the repository
3. Submit pull requests

## Getting Help

- **GitHub Issues**: Create an issue on the repository
- **Documentation**: Check the [docs](../) directory
- **Examples**: Run the provided [examples](../examples/)

---

**Quick Start Time**: 10-15 minutes  
**Expected Outcome**: Running demos and basic usage  
**Next**: Explore advanced features and customization