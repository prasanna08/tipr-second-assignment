from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import Adam
from keras.callbacks import History
import json

logger = History()

# Model
model = Sequential()
model.add(Dense(500, input_shape=(784,), init='uniform', activation='tanh'))
model.add(Dense(250, init='uniform', activation='tanh'))
model.add(Dense(2))
model.add(Activation('softmax'))
opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

logs = model.fit(x=data_x, y=data_y, batch_size=2000, epochs=50, validation_split=0.8, callbacks=[logger])

f = open('keras-cat-dog-tanh.json', 'w')
f.write(json.dumps(logs.history))
f.close()
