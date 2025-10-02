# PyTorch Multi-Class Animal Image Classification

This repository contains the code for a multi-class image classification model built with PyTorch. The model is trained to classify images into three categories: **dogs**, **cats**, and **goats**.

This project follows a modular programming structure, making the code reusable and easy to understand. It also incorporates TensorBoard for experiment tracking and visualization.

## Live Demo

**(Placeholder)** Once you deploy your model, you can link to it here!
* **[Link to Live Demo on Hugging Face Spaces]**

## Features

* **Modular Codebase:** The project is broken down into separate Python scripts for data setup, model building, and training logic.
    * `data_setup.py`: Creates `DataLoader`s for the image data.
    * `model_builder.py`: Defines the CNN architecture (e.g., TinyVGG).
    * `engine.py`: Contains reusable functions for training and testing loops.
    * `train.py`: The main script to run the entire training process.
    * `utils.py`: Contains helper functions, such as for saving the model.
* **Experiment Tracking:** Integrated with **TensorBoard** to log and visualize metrics like loss and accuracy, as well as the model graph.
* **Custom Data:** The model is trained on a custom dataset of animal images downloaded from the web.

## Directory Structure

```
├── data/
│   ├── animals/
│   │   ├── train/
│   │   │   ├── dogs/
│   │   │   ├── cats/
│   │   │   └── goats/
│   │   └── test/
│   │       ├── dogs/
│   │       ├── cats/
│   │       └── goats/
├── models/
│   └── (Saved model checkpoints like `tiny_vgg_model.pth` will appear here)
├── runs/
│   └── (TensorBoard log files will appear here)
├── going_modular/
│   ├── __init__.py
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── train.py
│   └── utils.py
└── requirements.txt
```

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8+
* PyTorch
* `torchvision`
* `matplotlib`
* `scikit-learn`
* `tensorboard`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone (https://github.com/sethiritham/pytorch_learning)
    cd pytorch_learning
    ```

2.  **Create a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    (First, you'll need to create the `requirements.txt` file by running `pip freeze > requirements.txt` in your terminal after installing the packages above).
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare the dataset:** The script in `pytorch_custom_datasets.ipynb` can be used to download and create the `data/animals` directory. Ensure this is done before training.

## Usage

All scripts should be run from the root project directory (the one containing the `going_modular` folder).

### Training the Model

To start training the model, run the `train.py` script as a module:

```bash
python -m going_modular.train
```
This will start the training process, save model checkpoints to the `models/` directory, and log results to the `runs/` directory.

### Monitoring with TensorBoard

To view the training progress, loss curves, and model graph:

1.  Open a new terminal and navigate to the project's root directory.
2.  Run the following command:
    ```bash
    tensorboard --logdir=runs
    ```
3.  Open your web browser and go to `http://localhost:6006/`.

## Future Improvements

* Experiment with different model architectures from `torchvision.models` (e.g., ResNet, EfficientNet).
* Use a larger, more robust dataset (e.g., a subset of ImageNet).
* Implement more advanced data augmentation techniques.
* Fine-tune hyperparameters for better performance.

## Acknowledgements

This project was built while following the incredible [Zero to Mastery PyTorch for Deep Learning course](https://www.learnpytorch.io/) by Daniel Bourke.
