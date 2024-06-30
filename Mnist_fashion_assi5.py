#LOAD THE DATA 
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Normalization
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize the data
normalizer = Normalization(input_shape=(28, 28))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
normalizer.adapt(X_train)

# Reshape data to include the channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
#DESIGN ND CREATE NEURAL NETWORK
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.initializers import lecun_normal

# Set the random seed for reproducibility
tf.random.set_seed(123)
np.random.seed(123)

def create_model():
    model = Sequential([normalizer])
    model.add(Flatten(input_shape=(28, 28, 1)))
    
    # Add 100 hidden layers with 100 neurons each
    for _ in range(100):
        model.add(Dense(100, activation='selu', kernel_initializer=lecun_normal()))
    
    model.add(Dense(10, activation='softmax'))
    return model

model = create_model()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# STEP 3:Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

#STEP 4:  Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)
#Given the complexity of achieving 99% accuracy, it might be necessary to experiment with different configurations, such as increasing the number of epochs, using data augmentation, or introducing dropout layers to prevent overfitting.
def create_model_with_dropout():
    model = Sequential([normalizer])
    model.add(Flatten(input_shape=(28, 28, 1)))
    
    for _ in range(100):
        model.add(Dense(100, activation='selu', kernel_initializer=lecun_normal()))
        model.add(Dropout(0.2))
    
    model.add(Dense(10, activation='softmax'))
    return model

model_with_dropout = create_model_with_dropout()

# Compile the model with dropout
model_with_dropout.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                           loss='categorical_crossentropy', 
                           metrics=['accuracy'])

# Train the model with dropout
history_with_dropout = model_with_dropout.fit(X_train, y_train, epochs=7, batch_size=32, validation_split=0.2)

# Evaluate the model with dropout
test_loss_with_dropout, test_acc_with_dropout = model_with_dropout.evaluate(X_test, y_test)
print(f'Test accuracy with dropout: {test_acc_with_dropout}')


