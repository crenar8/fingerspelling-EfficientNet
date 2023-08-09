import cv2
from tensorflow import keras
import numpy as np
from collections import OrderedDict
from efficientnet.tfkeras import EfficientNetB0

# Load saved Model
model = keras.models.load_model('asl_fingerspelling_model.h5')

# Dictionary
class_mapping = OrderedDict([
    (0, 'a'),
    (1, 'b'),
    (2, 'c'),
    (3, 'd'),
    (4, 'e'),
    (5, 'f'),
    (6, 'g'),
    (7, 'h'),
    (8, 'i'),
    (9, 'k'),
    (10, 'l'),
    (11, 'm'),
    (12, 'n'),
    (13, 'o'),
    (14, 'p'),
    (15, 'q'),
    (16, 'r'),
    (17, 's'),
    (18, 't'),
    (19, 'u'),
    (20, 'v'),
    (21, 'w'),
    (22, 'x'),
    (23, 'y') ])

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    # Show the final output
    cv2.imshow('Webcam', frame)

    # Input listening
    key = cv2.waitKey(1)

    # If "S" is pressed, then take the pic as frame
    if key == ord('s'):
        # Image pre-elaboration in order to adapt it for my model
        frame = cv2.resize(frame, (200, 200))
        frame = frame.astype(np.float32) / 255.0

        input_image = np.expand_dims(frame, axis=0)

        prediction = model.predict(input_image)
        result = np.argmax(prediction)

        predicted_letter = class_mapping[result]

        print("Predicted Class:", predicted_letter)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
