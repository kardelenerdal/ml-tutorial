from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt('reconstruction.csv', delimiter=',')
nofRows, nofColumns = dataset.shape
nofInputs = nofColumns-1
nofTrainRows = (int)(nofRows*4/5)
nofTestRows = nofRows - nofTrainRows

X = dataset[:, 0:nofInputs]
train_x = X[0:nofTrainRows, :]
test_x = X[nofTrainRows:nofRows, :]
y = dataset[:, nofInputs]
train_y = y[0:nofTrainRows]
test_y = y[nofTrainRows:nofRows]

model = Sequential()
model.add(Dense(12, input_dim=nofInputs, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(test_x, test_y)
print('Accuracy: %.2f' % (accuracy*100))
