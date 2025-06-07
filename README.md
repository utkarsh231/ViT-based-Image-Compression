# Vision Transformer Image Compression

This project implements an image compression model using Vision Transformers (ViT) with entropy modeling. The model is optimized for training on consumer GPUs and laptops.

## Features

- Vision Transformer-based image compression
- Entropy modeling for efficient compression
- Mixed precision training for faster training on consumer GPUs
- Automatic mixed precision (AMP) support
- Experiment tracking with Weights & Biases
- TensorBoard visualization
- Checkpoint saving and resuming
- Optimized for laptop GPU training

## Requirements

- Python 3.8+
- PyTorch 2.1.0+
- CUDA-capable GPU (for training)
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vit-compression.git
cd vit-compression
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The code supports various datasets. For testing, you can use the Kodak dataset:

```bash
python prepare_kodak_data.py
```

This will download and prepare the Kodak dataset in the `data/` directory.

## Training

To start training:

```bash
python train.py
```

The training script includes:
- Automatic mixed precision training
- Learning rate warmup
- Cosine learning rate scheduling
- Gradient clipping
- Checkpoint saving
- Experiment tracking

### Monitoring Training

1. TensorBoard:
```bash
tensorboard --logdir logs
```

2. Weights & Biases:
- The training script will provide a URL to view training progress
- Metrics are automatically logged

## Model Architecture

The model uses a Vision Transformer architecture with:
- Patch embedding
- Transformer encoder
- Entropy modeling
- Learned compression

Key components:
- `ViTEncoder`: Vision Transformer encoder
- `ContextAdaptiveEntropyModel`: Entropy modeling
- `SimpleViTDecoder`: Image reconstruction

## Configuration

The model can be configured through `config.py`. Key parameters:
- `img_size`: Input image size (default: 256)
- `patch_size`: Size of image patches (default: 16)
- `embed_dim`: Transformer embedding dimension (default: 384)
- `depth`: Number of transformer layers (default: 6)
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Initial learning rate (default: 2e-4)

## Performance

Expected performance on a laptop GPU:
- Training speed: ~2-3 images/second
- Memory usage: ~4-6GB GPU memory
- Training time: ~2-3 hours for 50 epochs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the Vision Transformer architecture
- Uses ideas from various image compression papers
- Inspired by the CompressAI library

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 