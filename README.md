# Facial Landmarks Detection

This project implements a deep learning pipeline for detecting facial landmarks using PyTorch Lightning. The model is trained on the iBUG 300-W dataset and predicts 68 facial landmarks for grayscale images.

## Features
- Custom dataset class for loading and preprocessing the iBUG 300-W dataset.
- ResNet18-based model for landmark detection.
- Training, validation, and testing pipelines with PyTorch Lightning.
- Visualization of predictions and ground truth landmarks.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- PyTorch and PyTorch Lightning
- Additional libraries: `torchvision`, `numpy`, `opencv-python`, `matplotlib`, `pillow`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hoanglmv/dt_landmarks.git
   cd dt_landmarks
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the iBUG 300-W dataset and place it in the appropriate directory.

## Usage

### Training the Model
Run can run the jupyter notebook `train.ipynb` to train the model. The notebook includes sections for:
- Loading the dataset
- Preprocessing images and landmarks
- Defining the model architecture
- Training and validation loops
- Saving the trained model

### Visualizing Predictions
After training, you can visualize the predictions on test images. Example below:
![Ảnh demo](demo.png)


### Model Checkpoints
The trained model is saved in the `trained_models` directory as `facial_landmarks_model.pth`.

## Results
The model predicts 68 facial landmarks with reasonable accuracy. Example visualizations are included in the notebook.

## License
This project is licensed under the MIT License.
