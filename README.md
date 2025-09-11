# PUMB: Parameter Update Memory Bank for Federated Learning

## 🚀 Overview

Advanced federated learning system with intelligent client selection, memory-based pattern recognition, and communication optimization.

## ✨ Key Features

- 🧠 **Memory Bank System**: FAISS-based storage of client update patterns
- 📊 **Enhanced Quality Metrics**: Multi-factor client assessment (loss, consistency, data)
- 🎯 **Intelligent Selection**: Theory-aligned exploration/exploitation balance
- 📡 **Communication Optimization**: ~50% reduction through client-side embeddings
- 🔄 **Adaptive Learning**: Dynamic quality metric adaptation across training phases

## 🛠️ Installation

```bash
git clone https://github.com/mohdlakas/QualityFL.git
cd QualityFL
pip install -r requirements.txt
pip install faiss-cpu  # or faiss-gpu for GPU support
```

## 🚀 Quick Start

```bash
cd src
python federated_pumb_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.1 --local_ep=5 --local_bs=32 --lr=0.0005 --iid=0 --alpha=0.5
```

## 📊 Performance Results

| Dataset | Distribution | Test Accuracy | Communication Reduction |
|---------|-------------|---------------|------------------------|
| CIFAR-10 | Non-IID (α=0.5) | ~85% | 50% |
| CIFAR-100 | Non-IID (α=0.1) | ~63% | 50% |
| MNIST | Non-IID (α=0.5) | ~99% | 50% |

## 🏗️ System Architecture

```
Client Side                     Server Side
┌─────────────────┐            ┌──────────────────┐
│ Local Training  │            │ Memory Bank      │
│ Embedding Gen   │ =========> │ FAISS Index      │
│ Statistics Comp │            │ Quality Calc     │
│ Update Tracking │            │ Client Selector  │
└─────────────────┘            └──────────────────┘
```

## 📁 Project Structure

```
src/
├── federated_pumb_main.py      # Main training script
├── federated_PUMB.py           # PUMB server implementation  
├── embedding_generator.py      # Parameter embedding generation
├── memory_bank.py              # FAISS-based memory system
├── intelligent_selector.py     # Client selection algorithms
├── quality_metric2.py          # Quality assessment metrics
├── update.py                   # Optimized local client updates
├── models.py                   # Neural network architectures
├── options.py                  # Command line arguments
├── utils_dir.py                # Utility functions
├── sampling_dir.py             # Data distribution functions
├── Algorithms/                 # Comparison algorithms (FedAvg, FedProx, etc.)
└── analysis/                   # Analysis and visualization tools
```

## ⚙️ Key Parameters

### Data Distribution
```bash
--iid=0 --alpha=0.5              # Non-IID Dirichlet distribution
--num_users=100 --frac=0.1       # 100 clients, select 10 per round
```

### Training Configuration
```bash
--epochs=100 --local_ep=5        # 100 global rounds, 5 local epochs
--lr=0.0005 --optimizer=adam     # Learning rate and optimizer
--local_bs=32                    # Local batch size
```

### PUMB Settings
```bash
--pumb_exploration_ratio=0.5     # Balance exploration vs exploitation
--pumb_initial_rounds=10         # Pure exploration phase
```

## 🔬 Core Innovations

### 1. Memory Bank System
- **FAISS Integration**: Efficient similarity search for 1000+ clients
- **Embedding Storage**: 512-dimensional parameter update representations
- **Pattern Recognition**: Historical client behavior analysis

### 2. Communication Optimization
- **Client-Side Processing**: Embeddings generated locally
- **Dual-Mode Operation**: Full embeddings (2KB) or statistics (50 bytes)
- **Redundancy Elimination**: No duplicate parameter transmission

### 3. Quality Assessment Framework
```python
Quality = α·Loss_Quality + β·Consistency_Quality + γ·Data_Quality
```
- **Loss Quality**: Relative improvement vs other clients
- **Consistency Quality**: Similarity to historical patterns  
- **Data Quality**: Contribution weight based on data size

### 4. Intelligent Client Selection
- **Theory-Aligned**: Balances exploration and exploitation
- **Reliability Tracking**: Long-term client performance assessment
- **Adaptive Thresholds**: Dynamic quality requirements

## 📈 Experimental Features

### Algorithm Comparisons
- **FedAvg**: Baseline federated averaging
- **FedProx**: Proximal term regularization
- **SCAFFOLD**: Variance reduction
- **FedNova**: Normalized averaging
- **Power of Choice**: Random client selection variants

### Analysis Tools
- **Comprehensive Logging**: Detailed training metrics
- **Visualization**: Training plots and client analysis
- **Hyperparameter Tuning**: Optuna-based optimization
- **Performance Comparison**: Multi-algorithm benchmarking

## 🎯 Results and Analysis

### Automatic Output
- `save/logs/`: Training logs and comprehensive analysis
- `save/images/`: Performance plots and visualizations
- `save/objects/`: Model checkpoints and metrics

### Key Metrics Tracked
- Test accuracy per round
- Communication overhead
- Client selection patterns
- Quality score distributions
- Memory bank evolution

## 🚀 Advanced Usage

### Custom Quality Metrics
```python
# In quality_metric2.py
server = PUMBFederatedServer(
    model, optimizer, loss_fn, args,
    quality_metric='stable'  # or 'generous', 'conservative'
)
```

### Memory Bank Configuration
```python
# Adjust memory capacity and embedding dimensions
memory_bank = MemoryBank(
    embedding_dim=512,
    max_memories=1000
)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📧 Contact

**Mohamed Lakas**
- Email: mohdlakas@gmail.com
- GitHub: [@mohdlakas](https://github.com/mohdlakas)

## 🙏 Acknowledgments

- PyTorch team for deep learning framework
- FAISS team for similarity search capabilities
- Federated learning research community
