import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

# Ensure reproducibility
tf.random.set_seed(123)

# Load the data
mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data to add a single channel (grayscale)
X_train_full = X_train_full[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# Split the full training data into training and validation sets
X_train, X_val = X_train_full[:50000], X_train_full[50000:]
y_train, y_val = y_train_full[:50000], y_train_full[50000:]
# Define the model
model = Sequential([
    # First layer
    Conv2D(64, kernel_size=7, padding='same', activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)),
    MaxPooling2D(),
    
    # Second layer
    Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
    Conv2D(128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D(),
    
    # Third layer
    Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
    Conv2D(256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D(),
    
    # Flatten and dense layers
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
# Test the model on the first 20 examples of the test data
predictions = model.predict(X_test[:20])
predicted_labels = tf.argmax(predictions, axis=1)
actual_labels = y_test[:20]

# Calculate accuracy on the first 20 examples
accuracy_20 = tf.reduce_mean(tf.cast(predicted_labels == actual_labels, tf.float32)).numpy()
print(f'Accuracy on the first 20 examples: {accuracy_20}')
