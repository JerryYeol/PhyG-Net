# PhyG-Net

# Dual-Modal Pretraining and Finetuning Framework

This repository implements a robust pretraining and finetuning framework designed for dual-modal data. The project leverages a contrastive learning architecture (incorporating window alignment and channel fusion modes) and supports adaptive model finetuning for few-shot downstream tasks. A supervised baseline trained from scratch is also included for performance comparison.

## 📂 Project Structure

The project follows a modular design for maintainability and scalability:

```text
├── configs/            # Global configuration (hyperparameters, paths, device settings)
├── datasets/           # Data loading & preprocessing (DualModalDataset, few-shot splits)
├── models/             # Neural network architectures (DualModalPretainModel, FinetuneModel)
├── utils/              # Helper functions, custom losses (e.g., NTXentLoss, metrics)
├── main.py             # Main execution script (training/evaluation loops)
└── README.md           # Project documentation
```

## 📦 Data Preparation

The dataset required for this project is open-sourced and hosted on **Zenodo**. Before running the code, please download and set up the dataset.

**Setup Steps:**

1. Access the dataset via the Zenodo link and download the archive: [https://doi.org/10.5281/zenodo.19453692](https://doi.org/10.5281/zenodo.19453692)
2. Create a folder named `data` in the root directory of your project.
3. **Extract the downloaded dataset** directly into this `data` folder.

Your project directory should look like this:
```text
PhyG-Net/
  ├── data/           <-- Place your extracted train/test data here
  │   ├── train.xxx
  │   └── test.xxx
  ├── configs/
  ...
```
*(Note: Ensure that the extracted filenames match the `Config.TRAIN_FILE` and `Config.TEST_FILE` paths defined in `configs/config.py`.)*

## ⚙️ Dependencies

It is recommended to use Python 3.8+. Install the required dependencies using pip:

```bash
pip install torch torchvision numpy scikit-learn tqdm
pip install -r requirements.txt
```

## 🚀 Usage

The core execution flow is managed within `main.py`. You can seamlessly switch between different experimental modes (Pre-training, Fine-tuning, and Baseline).

### 1. Quick Start

By default, executing the main script will run the active pipeline defined at the bottom of `main.py`:

```bash
python main.py
```

### 2. Execution Modes

Inside the `if __name__ == '__main__':` block in `main.py`, you can uncomment specific lines to trigger different training stages:

*   **Pre-training Mode (`run_pretrain`)**: 
    Performs self-supervised/unsupervised representation learning on dual-modal data. It optimizes the network using `NTXentLoss` (contrastive loss) and Cross-Entropy loss. Weights are saved to `Config.CKPT_PATH`.
    
*   **Fine-tuning Mode (`run_finetune`)**: 
    Loads the pretrained checkpoint and evaluates the model on a few-shot dataset. It utilizes a **Partial Freezing strategy** and **Differential Learning Rates** (e.g., $1e-4$ for the encoder, $1e-3$ for the classifier).
    
*   **Supervised Baseline (`run_baseline`)**: 
    Initializes the model with random weights and trains from scratch using the few-shot dataset. This serves as a control group to demonstrate the effectiveness of the pretraining phase.

### 3. Configuration

Hyperparameters and global settings can be easily modified in `configs/config.py`. Key parameters include:
*   `DEVICE`: Hardware acceleration (`cuda` or `cpu`)
*   `PRETRAIN_EPOCHS` & `FINETUNE_EPOCHS`: Number of training epochs
*   `RATIO`: The sample ratio for Few-shot dataset generation
*   `D_MODEL`: The hidden feature dimension of the models
*   Path settings and learning rates.

## 📊 Evaluation Metrics

The training and evaluation loops will automatically compute and display progress via `tqdm`. The primary metrics tracked are:
*   **Accuracy (ACC)**
*   **Weighted F1-Score (F1)**

Best performing metrics will be updated and displayed in real-time during the evaluation phase of each epoch.

## 📝 Citation

If you use the benchmark dataset or framework in your research, please cite the Zenodo repository using the DOI below:

**DOI:** `10.5281/zenodo.18933365`

```bibtex
@dataset{your_dataset_name,
  author       = {Your Name/Organization},
  title        = {Your Dataset Title},
  month        = {Month},
  year         = {Year},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18933365},
  url          = {https://doi.org/10.5281/zenodo.18933365}
}
