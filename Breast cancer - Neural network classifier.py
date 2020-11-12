import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Prepare dataset for training
df = pd.read_csv("data.csv", sep = ',')
a = df.replace(['M','B'], [1,0])
a = a.dropna(axis = 1)
train, test = train_test_split(a.values, test_size = 0.30)
y_train, y_test = train[:, 1], test[:, 1]
x_train, x_test = np.delete(train, [0,1], axis = 1), np.delete(test, [0,1], axis = 1)
# Fix the seed
np.random.seed(1)
# Create model
model = Sequential()
model.add(Dense(12, input_dim=x_train.shape[1], activation='tanh'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(x_train, y_train, validation_split = 0.20, epochs=100, batch_size=9)
# Evaluate the model
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f" % (model.metrics_names[0], scores[0]))
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()