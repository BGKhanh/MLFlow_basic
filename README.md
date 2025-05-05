# CIFAR-10 Training Pipeline with Multiple Models

This repository contains a complete machine learning pipeline for training a multiple of pytorch pretrained model on the CIFAR-10 and CIFAR-100 datasets. The pipeline includes data preprocessing, model training, validation, evaluation, hyperparameter tuning, and distributed training capabilities.

## Key Features

- **End-to-end Training Pipeline**: Automated workflow from data preprocessing to model evaluation
- **Experiment Tracking**: Comprehensive tracking of all runs with MLflow
- **Hyperparameter Optimization**: Automated tuning with Ray Tune
- **Distributed Training**: Parallel processing using Ray for faster training
- **Multiple Optimizers**: Support for Adam, AdamW, SGD, and L-BFGS optimizers
- **Multiple Models**: Support for MobileNet, EfficientNet, DenseNet, and RegNet architectures
- **Reproducible Environment**: Specific dependency versions for consistent results

## Technologies Used

- **PyTorch & torchvision**: For model development and training
- **Albumentations**: For advanced data augmentation
- **MLflow**: For experiment tracking and model management
- **Ray & Ray Tune**: For distributed training and hyperparameter optimization

## Project Structure

```
├── config/                  # Configuration files
│   └── config.yaml          # Main configuration
├── src/                     # Source code
│   ├── data/                # Data processing scripts
│   ├── models/              # Model definitions
│   ├── training/            # Training scripts
│   ├── utils/               # Utility functions
│   └── main.py              # Main entry point
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Setup

### Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run MLFlow UI:
   ```bash
   mlflow ui --port 5000
   ```

## Running the Pipeline

### Basic Usage

To run the standard training pipeline with default settings:

```bash
python main.py
```

### Advanced Usage

The pipeline can be run in different modes by modifying the command line arguments:

```bash
# For hyperparameter tuning
python main.py --tune

# For distributed training
python main.py --distributed

# For distributed training with hyperparameter tuning first
python main.py --tune --distributed
```

### Configuration

All parameters are defined in `config/config.yaml`. You can modify this file to change:

1. **Dataset Configuration**:
   - Choose between CIFAR-10 and CIFAR-100
   - Adjust image size and data loading parameters

2. **Model Configuration**:
   - Select model architecture (mobilenet, efficientnet, densenet, regnet)
   - Enable/disable pretrained weights
   - Configure dropout rate

3. **Training Configuration**:
   - Adjust learning rate, weight decay, batch size
   - Set early stopping parameters

4. **Augmentation Configuration**:
   - Enable/disable various augmentation techniques
   - Configure augmentation intensity

5. **Distributed Training Configuration**:
   - Set world size and backend
   - Configure resource allocation
   - Configure training device

Example of changing model and optimizer in config.yaml:
```yaml
# Model configuration
model:
  name: "efficientnet"  # Options: "mobilenet", "efficientnet", "densenet", "regnet"
  pretrained: true
  dropout_rate: 0.2

# Training configuration
training:
  batch_size: 128
  num_epochs: 10
  checkpoint_dir: "checkpoints"
  optimizer: "adamw"  # Only AdamW is allowed
  learning_rate: 0.0001
  weight_decay: 0.01
  # Scheduler options
  scheduler: "cosine"  # Options: "step", "cosine", "none"
  step_size: 10
  gamma: 0.1
  early_stopping: true
  patience: 5
  seed: 42
  # Mixed precision training
  use_amp: true  # Enable automatic mixed precision training
```

## Pipeline Components

1. **Data Preprocessing**: 
   - CIFAR-10 and CIFAR-100 dataset loading
   - Normalization and augmentation using Albumentations
   - Data visualization capabilities

2. **Model Setup**:
   - Multiple models including MobileNet, EfficientNet, RegNet and DenseNet 
   - Classifier adjustment for CIFAR-10 classes and CIFAR-100

3. **Training & Validation**:
   - PyTorch training loop
   - Multiple optimizers including AdamW
   - Learning rate scheduling
   - CrossEntropyLoss as the loss function
   - Validation metrics tracking

4. **Experiment Tracking**:
   - MLflow logging of all hyperparameters
   - Dataset metadata tracking
   - Performance metrics recording
   - Model checkpoints and artifacts storage

5. **Hyperparameter Tuning**:
   - Ray Tune integration for automated optimization
   - Support for various search algorithms
   - Parallel execution of trials
   - Automatic optimization of optimizer choice and parameters

6. **Distributed Training**:
   - Ray-based distributed training
   - Resource allocation optimization
   - Synchronized model updates

