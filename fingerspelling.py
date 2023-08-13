import cv2
from tensorflow import keras
import numpy as np
from collections import OrderedDict
from efficientnet.tfkeras import EfficientNetB0

# Load saved Model
model = keras.models.load_model('asl_fingerspelling_model.h5')

# Dictionary
class_mapping = OrderedDict([
    (0, 'A'),
    (1, 'B'),
    (2, 'C'),
    (3, 'D'),
    (4, 'E'),
    (5, 'F'),
    (6, 'G'),
    (7, 'H'),
    (8, 'I'),
    (9, 'J'),
    (10, 'K'),
    (11, 'L'),
    (12, 'M'),
    (13, 'N'),
    (14, 'O'),
    (15, 'P'),
    (16, 'Q'),
    (17, 'R'),
    (18, 'S'),
    (19, 'T'),
    (20, 'U'),
    (21, 'V'),
    (22, 'W'),
    (23, 'X'),
    (24, 'Y'),
    (25, 'Z'),
    (26, 'del'),
    (27, 'nothing'),
    (28, 'space'), ])

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
