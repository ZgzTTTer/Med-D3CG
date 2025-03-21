# Med-D3CG: Wavelet-Based Diffusion in the Difference Domain for Cross-Modality Medical Image Generation
## 📌 Overview
**Med-D3CG (Differential Domain Medical Diffusion for Conditional Generation)** is a novel framework for cross-modal medical image synthesis. Instead of directly generating target images, Med-D3CG models the difference domain between conditioned and target images, capturing structural and intensity variations more effectively. The framework leverages Discrete Wavelet Transform (DWT) to enhance efficiency, accelerating the diffusion process while maintaining high image fidelity.

The overview of Med-D3CG:
![overview](static/images/overview.png)

The diffusion process:
![overview](static/images/process.png)

### ✨ Key Features
- **Difference Domain**: Captures residual information rather than directly synthesizing images.
- **Wavelet Module**: Uses DWT and IDWT to accelerate the diffusion process.
- **Supports Multiple Modalities**: Supports cross-modality translation.

  

## 🛠️ Installation
### Requirements
- Python 3.12+
- PyTorch 2.4.1+
- numpy==1.26.4
- pillow==10.4.0
- PyWavelets==1.8.0

### Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/midisec/Med-D3CG.git
cd Med-D3CG
pip install -r requirements.txt
```

---

## 🚀 Usage
We have prepared the SynthRAD2023 dataset that matches our code processing dataset, which is available at the link below:
```bash
https://drive.google.com/drive/folders/1TGIUDLbMxDao2o4o8zsxcY1xv5hSlwDK?usp=sharing
```


To prepare the dataset, organize your data into multiple subfolders, where each subfolder represents a single sample. Each subfolder should contain **two PNG images**:

- One **CT image** (filename starting with `"ct"`).
- One **MRI image** (filename starting with `"mri"`).

```
dataset/
│── sample1/
│   ├── ct_1.png  # ct image
│   ├── mri_1.png      # mri image
│
│── sample2/
│   ├── ct_2.png
│   ├── mri_2.png
│
│── sample3/
│   ├── ct_3.png
│   ├── mri_3.png
│
│── ...
```

### 1️⃣ Training the Model

```bash
python train.py --model_name D3CG --dataset_train_dir path/to/dataset --dataset_val_dir path/to/dataset --n_epochs 2000 --batch_size 2
```

### 2️⃣ Evaluating the Model

Make sure to set the model file, output folder, and test dataset folder before running the following command

```bash
python pinggu.py
```

### 3️⃣ Generating the Image

Pass in a .png image and select the generation model to generate the target image.

```bash
python generation.py
```

## 📜 Citation
If you find this work useful, please cite our paper:
```
...
```