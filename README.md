# ASL Fingerspelling Recognition - EfficientNet

This project utilizes Python and TensorFlow to create an EfficientNetB0 model for recognizing American Sign Language (ASL) alphabet signs. 
The model is then used in the `fingerspelling.py` file to classify webcam captures triggered by pressing the "s" key.

## Requirements

Before running the program, make sure you have the following requirements installed:

- Python 3.x: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- TensorFlow 2.x: [https://www.tensorflow.org/install](https://www.tensorflow.org/install)
- OpenCV: [https://pypi.org/project/opencv-python/](https://pypi.org/project/opencv-python/)

It is recommended to use a virtual environment to install and manage the project's dependencies.

## Installation

1. Clone the repository or download the source code: 
```
git clone https://github.com/crenar8/fingerspelling-EfficientNet.git
cd fingerspelling-EfficientNet
```

2. Install the required dependencies using pip:
```
pip install cv2
pip install numpy
pip install keras
pip install collections
pip install sklearn.metrics
pip install efficientnet.tfkeras
```


## Usage

To run the program and perform ASL alphabet sign recognition using the webcam, follow these steps:

1. Navigate to the project directory:
```
cd fingerspelling-EfficientNet
```

2. Run the `fingerspelling.py` script:
```
python fingerspelling.py
```

3. When the webcam feed appears, press the "s" key to capture an image for classification.

4. The program will use the EfficientNetB0 model to classify the captured image and display the predicted ASL alphabet sign on the screen.

![Instructions.png](..%2FInstructions.png)

## Customization

If you want to customize or train your own model, you can modify the code in the `fingerspelling_model_creation.py` file. Adjust the model architecture, hyperparameters, and training configurations according to your requirements.

## Contributing

Contributions to this project are welcome. If you find any issues or want to suggest enhancements, please create an issue or submit a pull request.

## License

This project is licensed under GNU License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- The ASL Alphabet dataset used for training the model was obtained from [source](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out).
- The EfficientNetB0 architecture was developed by Tan, M., Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."

## Contact

For any questions or inquiries, please contact [k.dervishi@students.uninettunouniversity.net](mailto:k.dervishi@students.uninettunouniversity.net)









