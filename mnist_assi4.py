# STEP 1: LOAD THE DATA
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # loading
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#STEP 2 : DESIGN ND CREATE MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization

def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(300, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(100, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    return model

model = create_model()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#STEP 3:TRAIN THE MODEL.
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
#STEP 4:EVALUATE THE MODEL
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)
#Step 5: Extend the Model to Have 3 Hidden Layers
def create_extended_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(300, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(200, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(100, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    return model

extended_model = create_extended_model()


extended_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the extended model
history = extended_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# Evaluate the extended model
test_loss, test_accuracy = extended_model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)    # print the accuracy



OUTPUT:
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step
Epoch 1/10
1500/1500 [==============================] - 19s 12ms/step - loss: 0.2318 - accuracy: 0.9312 - val_loss: 0.1228 - val_accuracy: 0.9624
Epoch 2/10
1500/1500 [==============================] - 13s 8ms/step - loss: 0.1132 - accuracy: 0.9647 - val_loss: 0.1038 - val_accuracy: 0.9679
Epoch 3/10
1500/1500 [==============================] - 13s 9ms/step - loss: 0.0881 - accuracy: 0.9727 - val_loss: 0.1005 - val_accuracy: 0.9716
Epoch 4/10
1500/1500 [==============================] - 12s 8ms/step - loss: 0.0671 - accuracy: 0.9787 - val_loss: 0.0900 - val_accuracy: 0.9741
Epoch 5/10
1500/1500 [==============================] - 13s 8ms/step - loss: 0.0578 - accuracy: 0.9810 - val_loss: 0.0945 - val_accuracy: 0.9723
Epoch 6/10
1500/1500 [==============================] - 13s 8ms/step - loss: 0.0499 - accuracy: 0.9836 - val_loss: 0.0958 - val_accuracy: 0.9732
Epoch 7/10
1500/1500 [==============================] - 12s 8ms/step - loss: 0.0435 - accuracy: 0.9856 - val_loss: 0.0838 - val_accuracy: 0.9770
Epoch 8/10
1500/1500 [==============================] - 12s 8ms/step - loss: 0.0384 - accuracy: 0.9864 - val_loss: 0.0924 - val_accuracy: 0.9756
Epoch 9/10
1500/1500 [==============================] - 13s 9ms/step - loss: 0.0367 - accuracy: 0.9874 - val_loss: 0.0905 - val_accuracy: 0.9765
Epoch 10/10
1500/1500 [==============================] - 12s 8ms/step - loss: 0.0310 - accuracy: 0.9896 - val_loss: 0.0950 - val_accuracy: 0.9760
313/313 [==============================] - 1s 3ms/step - loss: 0.0747 - accuracy: 0.9795
Test Accuracy: 0.9794999957084656
Epoch 1/10
1500/1500 [==============================] - 16s 9ms/step - loss: 0.2412 - accuracy: 0.9264 - val_loss: 0.1347 - val_accuracy: 0.9582
Epoch 2/10
1500/1500 [==============================] - 14s 9ms/step - loss: 0.1189 - accuracy: 0.9626 - val_loss: 0.1120 - val_accuracy: 0.9659
Epoch 3/10
1500/1500 [==============================] - 15s 10ms/step - loss: 0.0962 - accuracy: 0.9701 - val_loss: 0.1010 - val_accuracy: 0.9691
Epoch 4/10
1500/1500 [==============================] - 13s 9ms/step - loss: 0.0797 - accuracy: 0.9748 - val_loss: 0.0888 - val_accuracy: 0.9736
Epoch 5/10
1500/1500 [==============================] - 13s 9ms/step - loss: 0.0643 - accuracy: 0.9800 - val_loss: 0.0812 - val_accuracy: 0.9763
Epoch 6/10
1500/1500 [==============================] - 13s 9ms/step - loss: 0.0570 - accuracy: 0.9814 - val_loss: 0.0901 - val_accuracy: 0.9751
Epoch 7/10
1500/1500 [==============================] - 13s 9ms/step - loss: 0.0509 - accuracy: 0.9833 - val_loss: 0.0972 - val_accuracy: 0.9729
Epoch 8/10
1500/1500 [==============================] - 13s 9ms/step - loss: 0.0477 - accuracy: 0.9845 - val_loss: 0.0806 - val_accuracy: 0.9778
Epoch 9/10
1500/1500 [==============================] - 13s 9ms/step - loss: 0.0403 - accuracy: 0.9869 - val_loss: 0.0811 - val_accuracy: 0.9779
Epoch 10/10
1500/1500 [==============================] - 14s 9ms/step - loss: 0.0339 - accuracy: 0.9890 - val_loss: 0.0868 - val_accuracy: 0.9791
313/313 [==============================] - 1s 3ms/step - loss: 0.0766 - accuracy: 0.9802
Test Accuracy: 0.9801999926567078
