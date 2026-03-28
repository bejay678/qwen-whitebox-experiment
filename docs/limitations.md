# Limitations & Future Work

## ⚠️ Current Limitations

### 1. Scale Limitations
- **Validated Scale**: 256 vectors (128-dimensional)
- **Untested**: >10,000 vectors (performance scaling unknown)
- **Memory Usage**: FAISS index size grows linearly with vector count
- **Training Data**: Adapter trained on only 100 facts with 252 variations

### 2. Technical Limitations
- **Deletion Mechanism**: FAISS uses marking deletion (`IndexIDMap`); true deletion requires index rebuild
- **Update Overhead**: Updating a fact requires delete+add operations
- **C Adapter**: Currently CPU-only; no GPU/CUDA implementation
- **Batch Processing**: C adapter supports batch but not optimized for large batches

### 3. Integration Limitations
- **Knowledge Integration**: Qwen generation may sometimes "ignore" retrieved facts
- **Prompt Engineering**: Current prompt template is simple; may need optimization
- **Error Handling**: Limited error recovery in edge cases
- **Model Dependency**: Requires Qwen2.5-0.5B-Instruct model (954MB download)

### 4. Performance Limitations
- **GPU vs CPU**: PyTorch adapter on GPU may be faster than C adapter on CPU
- **Index Building**: FAISS IVF training time increases with vector count
- **Memory Footprint**: Full system requires ~2GB RAM (model + index + Python)

## 🔬 Experimental Constraints

### Training Constraints
- **Dataset Size**: 100 facts (limited diversity)
- **Training Time**: 11 seconds (may underfit with more data)
- **Validation**: Only internal validation; no external test set
- **Hyperparameters**: Fixed architecture; no extensive hyperparameter search

### Evaluation Constraints
- **Metrics**: Focused on retrieval speed and discriminative power
- **Comprehensive Evaluation**: Lacks end-to-end accuracy metrics
- **Comparative Baselines**: Compared to PyTorch brute-force only
- **Real-world Testing**: No deployment in production systems

## 🚀 Future Work

### Short-term Improvements (1-3 months)
1. **Scale Testing**
   - Extend to 10,000+ vectors
   - Measure performance scaling
   - Optimize FAISS parameters for larger scales

2. **GPU Acceleration**
   - Implement CUDA version of C adapter
   - Compare GPU vs CPU performance
   - Optimize memory access patterns

3. **Enhanced Editability**
   - Implement true deletion in FAISS
   - Add batch editing operations
   - Improve update efficiency

### Medium-term Goals (3-6 months)
1. **Automation Toolchain**
   - Automatic identification of white-boxable modules
   - One-click export from PyTorch to C
   - Automated testing and validation

2. **Multi-modal Extension**
   - Image memory modules
   - Audio memory modules
   - Cross-modal retrieval

3. **Production Optimization**
   - Reduce memory footprint
   - Improve latency under load
   - Add monitoring and logging

### Long-term Vision (6-12 months)
1. **Framework Development**
   - General white-boxing framework for LLMs
   - Support for multiple model architectures
   - Plugin system for different components

2. **Community & Ecosystem**
   - Standard benchmarks for white-boxing
   - Pre-trained white-box modules repository
   - Integration with popular ML frameworks

3. **Research Directions**
   - Theoretical analysis of white-boxing limits
   - Hybrid neural-symbolic architectures
   - Self-improving white-box systems

## 📊 Known Issues & Workarounds

### Issue 1: FAISS Index Size
**Problem**: Index grows with vector count  
**Workaround**: Use quantization (PQ, SQ) for large-scale deployment  
**Future Fix**: Implement hierarchical indexing

### Issue 2: C Adapter CPU-only
**Problem**: No GPU acceleration  
**Workaround**: Use PyTorch adapter on GPU for now  
**Future Fix**: CUDA implementation

### Issue 3: Knowledge Integration
**Problem**: LLM may ignore retrieved facts  
**Workaround**: Better prompt engineering, fact weighting  
**Future Fix**: Fine-tune LLM to better use retrieved facts

### Issue 4: Model Dependency
**Problem**: Requires specific Qwen model  
**Workaround**: Provide download script  
**Future Fix**: Support multiple model architectures

## 🧪 Reproducibility Notes

### Exact Environment
- Python 3.10.14
- PyTorch 2.5.1 with CUDA 12.1
- FAISS 1.13.2 (CPU)
- Transformers 5.4.0

### Hardware Requirements
- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB RAM, 5GB disk space, GPU for Qwen model
- **Tested On**: AutoDL RTX 4090D, 31GB RAM

### Known Compatibility Issues
1. **Windows**: C adapter compilation may require MinGW
2. **macOS**: FAISS may require manual compilation
3. **ARM Linux**: Some dependencies may need source build

## 🔍 Validation Status

### ✅ Fully Validated
- 80× retrieval acceleration (FAISS vs PyTorch)
- 275% discriminative power improvement
- C adapter numerical consistency
- Editable knowledge base operations

### ⚠️ Partially Validated
- Scale beyond 256 vectors
- End-to-end accuracy metrics
- Production deployment scenarios
- Multi-user concurrent access

### ❌ Not Yet Validated
- 1,000,000+ vector scale
- Real-world production workloads
- Long-term stability
- Security and privacy aspects

## 🤝 Contributing to Limitations

We welcome contributions to address these limitations! Areas where help is especially needed:

1. **Performance Optimization**: GPU implementations, batch processing
2. **Scale Testing**: Large-scale experiments and optimizations
3. **Integration**: Better LLM integration, prompt engineering
4. **Tooling**: Automation, monitoring, deployment tools

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to contribute.

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-28  
**Status**: Active - limitations being addressed  
**Contact**: GitHub Issues for discussion