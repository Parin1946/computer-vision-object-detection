# Computer Vision Object Detection

A computer vision project focused on object detection using popular frameworks like YOLO and Faster R-CNN with PyTorch. Includes custom dataset preparation and model training.

## Project Overview

This repository delves into the exciting field of object detection within computer vision. It provides implementations and tutorials for various state-of-the-art object detection models, such as YOLO (You Only Look Once) and Faster R-CNN, primarily using the PyTorch deep learning framework. The project covers the entire pipeline from custom dataset annotation and preparation to model training, evaluation, and deployment.

## Features

-   **Multiple Models**: Implementations of YOLO (v3, v4, v5) and Faster R-CNN.
-   **Custom Dataset Training**: Guidelines and scripts for preparing and training models on your own datasets.
-   **Data Augmentation**: Techniques to enhance dataset diversity and model robustness.
-   **Performance Metrics**: Evaluation using metrics like mAP (mean Average Precision).
-   **Inference & Visualization**: Scripts for running inference on new images/videos and visualizing detection results.

## Getting Started

### Prerequisites

-   Python 3.8+
-   PyTorch
-   torchvision
-   OpenCV
-   Pillow
-   NumPy
-   Matplotlib

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Parin1946/computer-vision-object-detection.git
    cd computer-vision-object-detection
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To train a YOLOv5 model on a custom dataset:

```bash
python train_yolov5.py --data data/custom_dataset.yaml --epochs 50 --batch-size 16
```

To run inference with a trained model:

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source data/images/test_image.jpg
```

Refer to the specific model directories for detailed instructions.

## Contributing

Contributions are highly encouraged! Please open an issue for bugs or feature requests, or submit pull requests with new models, datasets, or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
