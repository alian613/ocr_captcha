# OCR CAPTCHA

This project is an OCR (Optical Character Recognition) tool for CAPTCHA verification based on PyTorch.   
It supports **training** and **evaluating** CAPTCHA recognition models and provides a simple CLI tool for generating images, training models, prediction, and evaluation.

📖 [中文說明文件](README_zh.md)

## ✨ Features

- ✅ Supports CAPTCHA image generation
- ✅ Supports CUDA/GPU acceleration (automatically uses GPU if CUDA is available)
- ✅ Supports recognition using trained models
- ✅ Customizable character set, CAPTCHA length, image size, and various training parameters
- ✅ Supports prediction on single image or batch of images in a folder
- ✅ Model evaluation function to calculate accuracy
- ✅ Simple CLI interface for ease of use

---

## 💾 Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Generate CAPTCHA Images
   ```bash
   python generate.py
   ```

   Or use the CLI tool:
   ```base
   python main.py generate
   ```
   
   > Use --help to view available options

### 2. Train Model
   ```bash
   python train.py
   ```

   Or use the CLI tool:
   ```base
   python main.py train
   ```

   > Use --help to view available options


### 3. Predict CAPTCHA
   Place images to be recognized in `./data/pred`
   ```bash
   python predict.py ./data/pred
   ```

   Or use the CLI tool:
   ```base
   python main.py predict ./data/pred
   ```

   > Use --help to view available options

---

## 📂 Project Structure
ocr-captcha/
├── cli.py              # CLI entry point  
├── config.py           # Configuration loader  
├── config.yaml         # Default configuration file  
├── train.py            # train the OCR model  
├── predict.py          # predict CAPTCHA from images  
├── evaluate.py         # Evaluates the model's accuracy  
├── codec.py            # Handles encoding/decoding of CAPTCHA strings to tensors and vice versa  
├── model.py            # Defines the CNN model architecture used for CAPTCHA recognition  
├── dataset.py          # Dataset loading and preprocessing (supports memory and on-disk modes)  
├── generate.py         # CAPTCHA image generator for training and testing  
├── requirements.txt    # Required dependencies  
├── data/               # Default folder for storing image datasets  
│   ├── eval/           # Images used for model evaluation (default)  
│   ├── pred/           # Images to be predicted (default)   
│   └── raw/            # Training images (default)  
└── model/              # Folder to save/load trained models (default)  

---

## 💡Tips
- Speed up training with `--cache`
> When running `train.py`, you can add the `--cache` option to load the dataset into memory (RAM),   
> which avoids reading from disk and speeds up training. Make sure your system has enough RAM available.

- Reference accuracy
> Training with `--dataset-size 100000 --epochs 30` can achieve over 99.5% accuracy on the same dataset. 
> However, accuracy may drop to around 80% when tested on a different CAPTCHA dataset.

- Beware of `--eval-acc-threshold`
> By default (as defined in `config.yaml`), the evaluation accuracy threshold is set to 75. 
> If a model evaluation result falls below this during training, 
> the current training will be skipped to save time and speed up convergence.

- Use `config.yaml` to set default parameters
> Paths, image size, CAPTCHA length, charset, and more can be configured in `config.yaml`. 
> If CLI parameters are not explicitly specified, the values from `config.yaml` will be used.


---


## 🙋‍♀️ Contact
If you have any questions or suggestions, feel free to open an issue on GitHub!


---


## ⭐ Support My Open Source Project
If you appreciate my work, consider ⭐ starring this repository or buying me a coffee to support development. 
Your support means a lot to me — thank you!

### [Ko-fi Support](https://ko-fi.com/alian613)


---


## 📄 License | 授權條款
This project is licensed under the MIT License.



