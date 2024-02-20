from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import os

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
df = pd.read_csv('MY_Data/fall_dataset98.csv')
data = df.values.tolist()
wbx = np.array(data)
print(wbx.shape)
X = wbx.reshape(1970, 20, 132)

actions = np.array(['down', 'up'])

label_map = {label: num for num, label in enumerate(actions)}

labels = []
for action in actions:
    for i in range(985):
        labels.append(label_map[action])

# print(np.array(labels).shape)
y = to_categorical(labels).astype(int)
# print(y)
# print(y.shape)
# labels_ds = tf.data.Dataset.from_tensor_slices(labels)
# data_ds = tf.data.Dataset.from_tensor_slices(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape)
# print(X_test)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(20, 132), recurrent_activation='sigmoid'))
model.add(LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.1))
model.add(LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.1))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=100, callbacks=[tb_callback])
model.summary()
model.save('action2.h5')