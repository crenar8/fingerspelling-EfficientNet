from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

# Carica il modello pre-addestrato
model = keras.models.load_model('asl_fingerspelling_model.h5')

# Valuta il modello sui dati di test
test_directory = 'dataset/test'
batch_size = 32
image_size = (64, 64)

# Creazione del generatore di dati di test
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Valutazione del modello sui dati di test
loss, accuracy = model.evaluate(test_generator)

# Stampa delle metriche di valutazione
print('Loss:', loss)
print('Accuracy:', accuracy)
