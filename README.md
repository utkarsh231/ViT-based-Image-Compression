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

## Running on Google Colab

You can easily run this project on Google Colab using the provided `main.ipynb` notebook. This is perfect for users who don't have access to a local GPU or want to experiment with the model quickly.

### Quick Start with Colab

1. **Open Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com)
   - Sign in with your Google account

2. **Upload the Notebook**:
   - Click "File" → "Upload notebook"
   - Upload the `main.ipynb` file from this repository
   - OR clone the repository directly in Colab

3. **Enable GPU**:
   - Click "Runtime" → "Change runtime type"
   - Select "GPU" as the Hardware accelerator
   - Click "Save"

4. **Run the Setup**:
   - Execute the cells in sequence
   - The notebook will automatically:
     - Clone the repository
     - Install dependencies
     - Mount Google Drive (optional)
     - Prepare datasets
     - Start training

### Colab Features

- **Automatic Setup**: No manual installation required
- **GPU Access**: Free GPU access for faster training
- **Google Drive Integration**: Save models and outputs to your Drive
- **Easy Testing**: Test the model on single images with built-in upload functionality
- **Progress Monitoring**: Real-time training progress with TensorBoard

### Important Notes for Colab Users

- **Free Tier Limitations**: 
  - Limited GPU hours (usually 12 hours per session)
  - Sessions may disconnect after inactivity
  - Consider Colab Pro for longer training sessions

- **Memory Management**:
  - Reduce batch size if you encounter out-of-memory errors
  - Default batch size is optimized for Colab's GPU memory

- **Saving Progress**:
  - Always save checkpoints to Google Drive
  - Use the provided save paths in the notebook
  - Download important results before session ends

### Testing on Single Images

The Colab notebook includes functionality to test the trained model on individual images:

1. Upload any image through the notebook interface
2. The model will compress and reconstruct the image
3. View comparison results with PSNR and BPP metrics
4. Download the results to your local machine

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