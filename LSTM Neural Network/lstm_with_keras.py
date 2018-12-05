from __future__ import print_function
import os
import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation

#based on coison.py Desktop/keras Toutorial
# Activate virtualenv (prompt will change)  ->  . ./.py35/bin/activate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

seed = 7
np.random.seed(seed)

with open('Train_X', 'rb') as fp:
    train_X = pickle.load(fp)
with open('Train_Y', 'rb') as fp:
    train_Y = pickle.load(fp)

with open('Test_X', 'rb') as fp:
    test_X = pickle.load(fp)
with open('Test_Y', 'rb') as fp:
    test_Y = pickle.load(fp)

n_iputs = 2
batch_size = 2
seq_len = 300
n_epochs = 1000

#afairw ta teleutea gia na orisw swsto batch_size
print(len(train_X))
print(len(test_X))
train_X = train_X[:-1] # all but last 
train_Y = train_Y[:-1] #all but last 

test_X = test_X[:-1] # all but last 
test_Y = test_Y[:-1] #all but last 


print('Creating Model...')

model = Sequential()

model.add(LSTM(50,
               input_shape=(n_iputs,seq_len),
               batch_size=batch_size,
               return_sequences=True,
               stateful=False))     # Boolean (default False). If True, the last state for each sample at
                                    # index i in a batch will be used as initial state for the sample of index i
                                    # in the following batch.
#model.add(Dropout(0.1))
model.add(Activation('sigmoid'))

model.add(LSTM(35,
               input_shape=(n_iputs,seq_len),
               batch_size=batch_size,
               return_sequences=False,
               stateful=False))
#model.add(Dropout(0.1))
model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation('linear'))

#Rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0)

model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['accuracy'])

print('Training...')

callbacks = [EarlyStopping(monitor='val_loss',patience=13, verbose=1)]
history = model.fit(train_X,train_Y,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    shuffle=True,
                    verbose=2,
                    validation_split=0.098,
                    callbacks=callbacks)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Saving Model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.model.save_weights("model.h5")
print("Saved model to disk")


print('Predicting')
predicted_output = model.predict(test_X, batch_size=batch_size)

real_energy = 0
for i in test_Y:
	for j in i:
		real_energy +=j

prediction = 0
for i in predicted_output:
	for j in i:
		prediction +=j

print("Real Energy = ",real_energy,"	Predicted Energy= ", prediction)

print('Plotting Results')
plt.subplot(2, 1, 1)
plt.plot(test_Y)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()

with open('prediction', 'wb') as fp:
    pickle.dump(predicted_output, fp)
with open('Real Energy', 'wb') as fp:
    pickle.dump(test_Y, fp)