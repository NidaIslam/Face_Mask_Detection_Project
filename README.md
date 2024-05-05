
# Face Mask Detection Using Convolutional Neural Networks

## Overview
This project aims to detect whether a person is wearing a face mask or not using deep learning techniques. It utilizes Convolutional Neural Networks (CNNs) to classify images into two classes: "With Mask" and "Without Mask".

## Installation
To run this project locally, follow these steps:

1. Install the required Python packages by running:

   pip install numpy matplotlib opencv-python pillow scikit-learn tensorflow
3. Clone this repository to your local machine:

   git clone https://github.com/nidaislam/face-mask-detection.git
5. Navigate to the project directory:

   cd face-mask-detection
7. Run the Python script:

   python mask_detection.py

## Dataset
The dataset used in this project consists of images of people with and without face masks. It is organized into two folders: "with_mask" and "without_mask".

Number of with mask images: 3725

Number of without mask images: 3828

## Preprocessing
Images are resized to (128, 128) pixels.

Images are converted to RGB format and converted into numpy arrays.

## Model Architecture
The CNN model consists of the following layers:

1. Convolutional layer with 32 filters and ReLU activation.
2. MaxPooling layer.
3. Convolutional layer with 64 filters and ReLU activation.
4. MaxPooling layer.
5. Flatten layer.
6. Dense layer with 128 units and ReLU activation.
7. Dropout layer with a dropout rate of 0.5.
8. Dense layer with 64 units and ReLU activation.
9. Dropout layer with a dropout rate of 0.5.
10. Output layer with 2 units and sigmoid activation.

## Training
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. The training data is split into training and validation sets with a ratio of 80:20.

## Results
Test Accuracy: 90.80%

## Predictive System
To use the predictive system:
Replace the input_image_path variable with the path to your image.

Run the code snippet to predict whether the person in the image is wearing a mask or not.

### Example Results
![image](https://github.com/NidaIslam/Face_Mask_Detection_Project/assets/121117931/7463d328-4899-4568-ad3a-729eec1eff2d)

Prediction: The person in the image is wearing a mask.

![image](https://github.com/NidaIslam/Face_Mask_Detection_Project/assets/121117931/f6f1101d-a47c-441e-8963-33aa1c2b914c)

Prediction: The person in the image is not wearing a mask.

## Dataset
You can download the dataset used in this project from [this link](https://drive.google.com/file/d/12C2H_XSYWJAu1zBtztkH57PsDrlj1TfY/view?usp=drive_link).


## Contact
For any questions, feel free to contact me at http://www.linkedin.com/in/nidaislam10.



