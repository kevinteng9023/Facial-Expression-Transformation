# Facial-Expression-Transformation with GAN

This project uses a GAN-based approach to transform facial images into representations of different emotions. The system processes input images, extracts faces, and generates transformed images with the specified emotion.

## Getting Started

### Prerequisites

Ensure you have the following modules installed in your Python environment:

```bash
pip install torch torchvision matplotlib numpy opencv-python mtcnn pillow
```

### File Structure

* `dataProcess.py`: Contains utilities for processing input images and extracting faces using the MTCNN face detection library.

* `myGAN.py`: Defines the architecture of the GAN, including the generator and discriminator.

* `Generate.py`: The main script to generate emotion-transformed images.

* `trainGAN.ipynb`: Jupyter Notebook for training the GAN to generate the desired weights for emotion transformation.

### Usage

1. Place the input images in the `./input` directory.

2. Run the following command to generate emotion-transformed images:

    ```bash
    python Generate.py
    ```
3. The transformed images will be saved in the `test_generated_images` directory.

### Emotion Options

The project supports transforming faces to the following emotions:

* Angry

* Disgust

* Fear

* Happy

* Sad

* Surprise

Modify the emotion in `Generate.py` to select the desired transformation:

```python
emotion_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
emotion = emotion_list[0]  # Choose an emotion from the list
```

### Output

* All the output directory will be generated automatically.

* Cropped and aligned faces are saved in the `./output_face` directory.

* Generated images are saved in the `test_generated_images` directory.

## Modules Overview

1. `dataProcess.py`

* Uses MTCNN for face detection and alignment.

* Preprocesses input images for further processing.

2. `myGAN.py`

* Defines the generator and discriminator for the GAN.

* The generator transforms input faces into the desired emotion.

3. `Generate.py`

* Loads the pre-trained generator model.

* Processes input images and generates emotion-transformed outputs.

4. `trainGAN.ipynb`

* Provides an interface for training the GAN.

* Allows users to configure training parameters and monitor progress.


## Pre-trained Models

Ensure you have the pre-trained GAN models for each emotion saved in the `saved_models` directory, named as `generator_<emotion>.pth` (e.g., `generator_happy.pth`).


## Training the GAN

To train the GAN to generate your desired weights, use the provided Jupyter Notebook:

### Prerequisites

Ensure that Jupyter Notebook is installed in your environment. You can install it using:

```bash
pip install notebook
```

### Training Steps

1. Open the trainGAN.ipynb notebook

2. Follow the steps in the notebook to:

* Set up your dataset in `data` directory.

* Configure hyperparameters (e.g., learning rate, batch size, epochs).

* Train the GAN to generate weights for specific emotions.

3. Once training is complete, the trained weights will be saved in the `saved_models` directory, named as `generator_<emotion>.pth` and `discriminator_<emotion>.pth`.

## Notes

* Ensure the input images are of high quality with visible faces for accurate detection and transformation.

* The script will automatically handle GPU acceleration if available.


### Enjoy generating emotion-transformed images!