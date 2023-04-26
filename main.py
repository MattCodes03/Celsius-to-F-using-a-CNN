import tensorflow as tf
import numpy as np

# Training Data
c = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)  # Degrees Celcius
f = np.array([-40,  14, 32, 46, 59, 72, 100],
             dtype=float)  # Degrees Fahrenheit

# CNN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=[1]),  # Input
    tf.keras.layers.Dense(units=4),  # Hidden
    tf.keras.layers.Dense(units=1),  # Output
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

# Training
training = model.fit(c, f, epochs=500, verbose=False)
print("Training Finished!")


def manualConversion(c):
    return (1.8*c) + 32


# Model prediction
print(f"Model: {model.predict([202.4])}")

# Manual Calculation
print(f"Manual Calculation: {manualConversion(202.4)}")
