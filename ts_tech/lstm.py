import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

time = np.arange(0, 10, 0.1)
object_position = 2 * time + 1  # Linear motion: position = 2*time + 1

object_position += np.random.normal(scale=0.5, size=len(object_position))

data = object_position[:-1].reshape(-1, 1, 1)
target = object_position[1:].reshape(-1, 1)  # Predict the next position

model = Sequential()
model.add(LSTM(units=20, activation='tanh', input_shape=(1, 1)))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(data, target, epochs=50, batch_size=1)

predictions = model.predict(data)

plt.figure(figsize=(10, 6))
plt.plot(time[:-1], target, label='Actual Position')
plt.plot(time[:-1], predictions, label='Predicted Position', linestyle='dashed')
plt.title('LSTM Prediction of Object Position Over Time')
plt.xlabel('Time')
plt.ylabel('Object Position')
plt.legend()
plt.show()

