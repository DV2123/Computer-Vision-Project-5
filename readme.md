# Project 5: Recognition Using Deep Networks

## Author
Divya

## Video Links
- Extension 2 (Live Video Digit Recognition): https://drive.google.com/file/d/1iI77E1l0X-Fd401YFDDWgRRz_0TpC-K3/view?usp=sharing

## Time Travel Days
No time travel days used.

## Custom Data Links
- Custom handwritten digits: included in `custom_digits/` folder
- Custom Greek letters: included in `custom_greek/` folder

## Files

### Task 1: Build and Train a CNN
- `train_mnist.py` - Builds, trains, and saves the MNIST CNN model
- `test_mnist.py` - Loads the model, tests on test set and custom handwritten digits

### Task 2: Examine the Network
- `examine_network.py` - Analyzes conv1 filters and visualizes their effects

### Task 3: Transfer Learning on Greek Letters
- `greek_transfer.py` - Adapts MNIST CNN for Greek letter recognition (alpha, beta, gamma)

### Task 4: Transformer Network
- `train_transformer.py` - Vision Transformer (ViT) implementation for MNIST digit recognition

### Task 5: Experiment
- `experiment.py` - Automated round-robin search over 3 dimensions (depth, heads, dropout) on Fashion MNIST using the transformer architecture. 24 runs total.

### Extensions
- `ext_pretrained_analysis.py` - Loads pre-trained VGG16 and visualizes its convolutional layers, compares with MNIST CNN filters
- `ext_live_recognition.py` - Real-time webcam digit recognition using the trained MNIST CNN

### Other Files
- `report.pdf` - Project report
- `mnist_model.pth` - Trained CNN model weights
- `transformer_model.pth` - Trained transformer model weights
- `greek_model.pth` - Trained Greek letter model weights
- `experiment_results.csv` - Task 5 experiment results

## How to Run
All code was developed and tested using Python 3.x with the `CVP5` conda environment.

```bash
conda activate CVP5

# Task 1: Train CNN
python train_mnist.py

# Task 1: Test CNN
python test_mnist.py

# Task 2: Examine network
python examine_network.py

# Task 3: Greek letter transfer learning
python greek_transfer.py

# Task 4: Train transformer
python train_transformer.py

# Task 5: Run experiment
python experiment.py

# Extension 1: VGG16 analysis
python ext_pretrained_analysis.py

# Extension 2: Live webcam recognition
python ext_live_recognition.py
```

## Dependencies
- PyTorch (torch)
- torchvision
- matplotlib
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
