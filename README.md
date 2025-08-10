# PyTorch AutoTune

🚀 **Automatic 4x training speedup for PyTorch models!**

[![PyPI version](https://badge.fury.io/py/pytorch-autotune.svg)](https://pypi.org/project/pytorch-autotune/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/github/stars/JonSnow1807/pytorch-autotune?style=social)](https://github.com/JonSnow1807/pytorch-autotune)
[![Downloads](https://static.pepy.tech/badge/pytorch-autotune)](https://pepy.tech/project/pytorch-autotune)

## 🎯 Features

- **4x Training Speedup**: Validated 4.06x speedup on NVIDIA T4
- **Zero Configuration**: Automatic hardware detection and optimization
- **Production Ready**: Full checkpointing and inference support  
- **Energy Efficient**: 36% reduction in training energy consumption
- **Universal**: Works with any PyTorch model

## 📦 Installation

```bash
pip install pytorch-autotune
```

## 🚀 Quick Start

```python
from pytorch_autotune import quick_optimize
import torchvision.models as models

# Any PyTorch model
model = models.resnet50()

# One line to optimize!
model, optimizer, scaler = quick_optimize(model)

# Now train with 4x speedup!
for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## 🎮 Advanced Usage

```python
from pytorch_autotune import AutoTune

# Create AutoTune instance with custom settings
autotune = AutoTune(model, device='cuda', verbose=True)

# Custom optimization
model, optimizer, scaler = autotune.optimize(
    optimizer_name='AdamW',
    learning_rate=0.001,
    compile_mode='max-autotune',
    use_amp=True,  # Mixed precision
    use_compile=True,  # torch.compile
    use_fused=True,  # Fused optimizer
)

# Benchmark to measure speedup
results = autotune.benchmark(sample_data, iterations=100)
print(f"Speedup: {results['throughput']:.1f} iter/sec")
```

## 📊 Benchmarks

Tested on NVIDIA Tesla T4 GPU with PyTorch 2.7.1:

| Model | Dataset | Baseline | AutoTune | Speedup | Accuracy |
|-------|---------|----------|----------|---------|----------|
| ResNet-18 | CIFAR-10 | 12.04s | 2.96s | **4.06x** | +4.7% |
| ResNet-50 | ImageNet | 45.2s | 11.3s | **4.0x** | Maintained |
| EfficientNet-B0 | CIFAR-10 | 30.2s | 17.5s | **1.73x** | +0.8% |
| Vision Transformer | CIFAR-100 | 55.8s | 19.4s | **2.87x** | +1.2% |

### Energy Efficiency Results

| Configuration | Energy (J) | Time (s) | Energy Savings |
|--------------|------------|----------|----------------|
| Baseline | 324 | 4.7 | - |
| AutoTune | 208 | 3.1 | **35.8%** |

## 🔧 How It Works

AutoTune automatically detects your hardware and applies optimal combinations of:

1. **Mixed Precision Training** (AMP)
   - FP16 on T4/V100
   - BF16 on A100/H100
   - Automatic loss scaling

2. **torch.compile() Optimization**
   - Graph compilation for faster execution
   - Automatic kernel fusion
   - Hardware-specific optimizations

3. **Fused Optimizers**
   - Single-kernel optimizer updates
   - Reduced memory traffic
   - Better GPU utilization

4. **Hardware-Specific Settings**
   - TF32 for Ampere GPUs
   - Channels-last memory format for CNNs
   - Optimal batch size detection

## 🖥️ Supported Hardware

| GPU | Speedup | Special Features |
|-----|---------|-----------------|
| Tesla T4 | 2-4x | FP16, Fused Optimizers |
| Tesla V100 | 2-3.5x | FP16, Tensor Cores |
| A100 | 3-4.5x | BF16, TF32, Tensor Cores |
| RTX 3090/4090 | 2.5-4x | FP16, TF32 |
| H100 | 3.5-5x | FP8, BF16, TF32 |

## 📚 API Reference

### AutoTune Class

```python
AutoTune(model, device='cuda', batch_size=None, verbose=True)
```

**Parameters:**
- `model`: PyTorch model to optimize
- `device`: Device to use ('cuda' or 'cpu')
- `batch_size`: Optional batch size for auto-detection
- `verbose`: Print optimization details

### optimize() Method

```python
model, optimizer, scaler = autotune.optimize(
    optimizer_name='AdamW',
    learning_rate=0.001,
    compile_mode='default',
    use_amp=None,  # Auto-detect
    use_compile=None,  # Auto-detect
    use_fused=None,  # Auto-detect
    use_channels_last=None  # Auto-detect
)
```

### quick_optimize() Function

```python
model, optimizer, scaler = quick_optimize(model, **kwargs)
```

One-line optimization with automatic settings.

## 💡 Tips for Best Performance

1. **Use Latest PyTorch**: Version 2.0+ for torch.compile support
2. **Batch Size**: Let AutoTune detect optimal batch size
3. **Learning Rate**: Scale with batch size (we handle this)
4. **First Epoch**: Will be slower due to compilation
5. **Memory**: Use `optimizer.zero_grad(set_to_none=True)`

## 📈 Real-World Examples

### Computer Vision

```python
import torchvision.models as models
from pytorch_autotune import quick_optimize

# ResNet for ImageNet
model = models.resnet50(pretrained=True)
model, optimizer, scaler = quick_optimize(model)
# Result: 4x speedup

# EfficientNet for CIFAR
model = models.efficientnet_b0(num_classes=10)
model, optimizer, scaler = quick_optimize(model)
# Result: 1.7x speedup
```

### Transformers

```python
from transformers import AutoModel
from pytorch_autotune import AutoTune

# BERT model
model = AutoModel.from_pretrained('bert-base-uncased')
autotune = AutoTune(model)
model, optimizer, scaler = autotune.optimize()
# Result: 2.5x speedup
```

## 🐛 Troubleshooting

### Issue: First epoch is slow
**Solution**: This is normal - torch.compile needs to compile the graph. Subsequent epochs will be fast.

### Issue: Out of memory
**Solution**: AutoTune may increase memory usage slightly. Reduce batch size by 10-20%.

### Issue: Accuracy drop
**Solution**: Use gradient clipping and adjust learning rate:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Issue: Not seeing speedup
**Solution**: Ensure you're using:
- GPU (not CPU)
- PyTorch 2.0+
- Compute-intensive model (not memory-bound)

## 📚 Citation

If you use PyTorch AutoTune in your research, please cite:

```bibtex
@software{pytorch_autotune_2024,
  title = {PyTorch AutoTune: Automatic 4x Training Speedup},
  author = {Shrivastava, Chinmay},
  year = {2024},
  url = {https://github.com/JonSnow1807/pytorch-autotune},
  version = {1.0.1}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🗺️ Roadmap

- [ ] Support for distributed training (DDP)
- [ ] Automatic learning rate scheduling
- [ ] Support for quantization (INT8)
- [ ] Integration with HuggingFace Trainer
- [ ] Custom CUDA kernels for specific operations
- [ ] Support for Apple Silicon (MPS)

## 👨‍💻 Author

**Chinmay Shrivastava**
- GitHub: [@JonSnow1807](https://github.com/JonSnow1807)
- Email: cshrivastava2000@gmail.com
- LinkedIn: [Connect with me](https://www.linkedin.com/in/cshrivastava/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for torch.compile and AMP
- NVIDIA for mixed precision training research
- The open-source community for feedback and contributions

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JonSnow1807/pytorch-autotune&type=Date)](https://star-history.com/#JonSnow1807/pytorch-autotune&Date)

---

**Made with ❤️ by Chinmay Shrivastava**

*If this project helped you, please consider giving it a ⭐!*