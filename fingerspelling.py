import cv2
from tensorflow import keras
import numpy as np
from collections import OrderedDict

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

    # Ascolta l'input da tastiera
    key = cv2.waitKey(1)

    # Se il tasto "S" viene premuto, scatta una foto
    if key == ord('s'):
        # Esegui le opportune operazioni di pre-elaborazione sull'immagine (ad esempio, ridimensionamento e normalizzazione)
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype(np.float32) / 255.0

        # Aggiungi una dimensione iniziale all'immagine per adeguarla all'input del modello
        input_image = np.expand_dims(frame, axis=0)

        prediction = model.predict(input_image)
        result = np.argmax(prediction)

        predicted_letter = class_mapping[result]

        print("Classe predetta:", predicted_letter)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
