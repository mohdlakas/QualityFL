# PUMB: Parameter Update Memory Bank for Federated Learning

## Overview

This project implements **PUMB (Parameter Update Memory Bank)**, an intelligent federated learning system that uses memory-based client selection and quality assessment to improve training efficiency and model performance.

## Key Features

- 🧠 **Memory Bank**: Stores client parameter update patterns using FAISS for similarity search
- 📊 **Quality Assessment**: Multi-factor quality metric (loss improvement, consistency, data contribution)
- 🎯 **Intelligent Selection**: Theory-aligned client selection with exploration/exploitation balance
- 📈 **Enhanced Communication**: ~50% reduction in communication overhead through client-side embedding generation
- 🔄 **Adaptive Learning**: Dynamic quality metric adaptation across training phases

## Architecture

```
Client Side:                    Server Side:
┌─────────────────┐            ┌──────────────────┐
│ Local Training  │            │ Memory Bank      │
│ Embedding Gen   │ =========> │ Quality Calc     │
│ Statistics Comp │            │ Client Selector  │
└─────────────────┘            └──────────────────┘
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YourUsername/PUMB-Federated-Learning.git
cd PUMB-Federated-Learning
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install FAISS (for similarity search):**
```bash
# CPU version
pip install faiss-cpu

# GPU version (if you have CUDA)
pip install faiss-gpu
```

## Quick Start

### Basic CIFAR-10 Training
```bash
cd src
python federated_pumb_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.1 --local_ep=5 --local_bs=32 --lr=0.0005 --iid=0 --alpha=0.5
```

### Key Parameters
- `--dataset`: Dataset (cifar, cifar100, mnist)
- `--iid`: 0 for Non-IID, 1 for IID
- `--alpha`: Dirichlet alpha for Non-IID distribution (lower = more heterogeneous)
- `--frac`: Fraction of clients selected per round
- `--local_ep`: Local training epochs
- `--pumb_exploration_ratio`: Exploration vs exploitation balance

## Performance Results

| Setting | Test Accuracy | Communication Reduction | Training Time |
|---------|--------------|------------------------|---------------|
| CIFAR-10 Non-IID (α=0.5) | 85.2% | ~50% | 42 min |
| CIFAR-100 Non-IID (α=0.1) | 62.8% | ~50% | 68 min |

## Project Structure

```
src/
├── federated_pumb_main.py      # Main training script
├── federated_PUMB.py           # PUMB server implementation
├── embedding_generator.py      # Parameter embedding generation
├── memory_bank.py              # FAISS-based memory system
├── intelligent_selector.py     # Client selection logic
├── quality_metric2.py          # Quality assessment metrics
├── update.py                   # Local client updates
├── models.py                   # Neural network models
├── options.py                  # Command line arguments
├── utils_dir.py                # Utility functions
└── sampling_dir.py             # Data distribution functions
```

## Key Innovations

### 1. Memory Bank System
- Stores client parameter update embeddings
- Uses FAISS for efficient similarity search
- Tracks client reliability and consistency

### 2. Enhanced Quality Metrics
- **Loss Quality**: Improvement relative to other clients
- **Consistency Quality**: Similarity to historical patterns
- **Data Quality**: Contribution based on data size

### 3. Communication Optimization
- Client-side embedding generation
- Statistical summary computation
- ~50% reduction in data transmission

## Configuration

### Quality Metric Parameters
```python
# In quality_metric2.py
class StableQualityMetric:
    def __init__(self, alpha=0.3, beta=0.2, gamma=0.5):
        # alpha: Loss weight
        # beta: Consistency weight  
        # gamma: Data weight
```

### Memory Bank Settings
```python
# In memory_bank.py
class MemoryBank:
    def __init__(self, embedding_dim=512, max_memories=1000):
        # embedding_dim: Dimension of parameter embeddings
        # max_memories: Maximum stored memory entries
```

## Experimental Settings

The project supports various experimental configurations:

- **Datasets**: CIFAR-10, CIFAR-100, MNIST
- **Data Distribution**: IID and Non-IID (Dirichlet)
- **Models**: CNN architectures
- **Optimizers**: SGD, Adam

## Results and Analysis

Results are automatically saved to:
- `save/logs/`: Comprehensive analysis reports
- `save/images/`: Training plots and visualizations
- `save/objects/`: Model checkpoints and metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pumb2024,
  title={PUMB: Parameter Update Memory Bank for Intelligent Federated Learning},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@YourUsername](https://github.com/YourUsername)