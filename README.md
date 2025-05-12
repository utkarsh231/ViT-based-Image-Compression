# ViT-based Image Compression

This repository contains a Vision Transformer (ViT)-based image compression pipeline implemented in PyTorch. The project demonstrates how to use transformer architectures for end-to-end learned image compression, including context-adaptive entropy modeling.

## Features
- Patch embedding and ViT encoder for image representation
- Context-adaptive entropy model (CAM) for learned compression
- Simple ViT-based decoder for image reconstruction
- Modular code for easy experimentation
- Training, validation, and inference scripts

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/utkarsh231/ViT-based-Image-Compression.git
   cd ViT-based-Image-Compression
   ```
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Dataset Preparation
- You can use your own dataset or the provided script to split a folder of images (e.g., cat images) into `data/train` and `data/val` with an 80:20 ratio:
  ```bash
  python prepare_cat_data.py
  ```
- For ImageNet-100 or other datasets, see the included preparation scripts.

## Training
Train the model using:
```bash
python train.py
```

## Inference
Run inference on a sample image:
```bash
python inference.py
```

## Customization
- Modify `config.py` to change model or training hyperparameters.
- See `dataset.py` for custom dataset loading.

## Repository Structure
- `tic_vit_encoder.py` — Main model architecture
- `entropy_model.py` — Context-adaptive entropy model
- `loss.py` — Loss functions
- `train.py` — Training script
- `inference.py` — Inference script
- `prepare_cat_data.py` — Example data preparation script
- `requirements.txt` — Python dependencies

## License
This project is for research and educational purposes.

---

Feel free to open issues or pull requests for improvements! 