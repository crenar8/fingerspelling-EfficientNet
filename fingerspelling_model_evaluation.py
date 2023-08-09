from keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB0
from sklearn.metrics import classification_report
from tensorflow import keras

# Loading of the pre-trained model
model = keras.models.load_model('asl_fingerspelling_model.h5')


test_directory = 'dataset/test'
batch_size = 32
image_size = (200, 200)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluation of the trained model
predictions = model.predict(test_generator)
y_pred = predictions.argmax(axis=1)
y_true = test_generator.classes

# Generate classification report
report = classification_report(y_true, y_pred)

# Print the report to a file
with open('model_evaluation_report.txt', 'w') as f:
    print(report, file=f)