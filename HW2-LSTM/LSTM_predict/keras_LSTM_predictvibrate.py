import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import SimpleRNN, Activation, Dense

SHIFT=5
TIME_STEPS = 50     # same as the height of the image
INPUT_SIZE = 3     # same as the width of the image
BATCH_SIZE = 50
OUTPUT_SIZE = 1
BATCH_START = 0
CELL_SIZE = 50
LR = 0.001
TRAIN_TEST_SEP = 45001
my_data = np.genfromtxt("outdata.csv", delimiter=',', dtype='float32')
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
print(my_data[:, 3].size)
X_train = my_data[1:TRAIN_TEST_SEP, 0:3]
X_test = my_data[TRAIN_TEST_SEP:66001, 0:3].reshape(
    (-1, TIME_STEPS, INPUT_SIZE))[0:50]
y_train = my_data[1:TRAIN_TEST_SEP, 3]
y_test = my_data[TRAIN_TEST_SEP:66001, 3].reshape(
    (-1, TIME_STEPS, 1))[0:50]

def get_batch():
    global BATCH_START, TIME_STEPS,SHIFT
    # xs shape (50batch, 20steps)
    X_batch = X_train[BATCH_START:BATCH_START + TIME_STEPS *
                      BATCH_SIZE,:].reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    Y_batch = y_train[BATCH_START:BATCH_START + TIME_STEPS *
                      BATCH_SIZE, ].reshape((BATCH_SIZE, TIME_STEPS, 1))
    if (BATCH_START + 2*TIME_STEPS * BATCH_SIZE) >= (X_train.shape[0]):
        BATCH_START=SHIFT
        SHIFT+=5
    else:
        BATCH_START += TIME_STEPS *BATCH_SIZE
    return [X_batch, Y_batch]


# def get_test():
#     X_batch = X_test[0:2500, :].reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
#     Y_batch = y_test[0:2500, ].reshape((BATCH_SIZE, TIME_STEPS, 1))
#     return [X_batch, Y_batch]

model = Sequential()
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),
    output_dim=CELL_SIZE,
    return_sequences=True,
    stateful=True
))

model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
model.add(Activation('sigmoid'))
adam=Adam(LR)
model.compile(optimizer=adam, loss='mse',
              metrics=['accuracy'])
print('Training ------------')

for step in range(500):
    X_batch, Y_batch= get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    if step % 10 == 0:
        #X_test, y_test=get_test()
        # X_test = X_test.astype('float32')
        # y_test = y_test.astype('int')
        #print(y_test.shape[0].dtype)
        cost ,accuracy= model.evaluate(X_test, y_test,batch_size=50, verbose=False)
        #error because we need a prediction each time step
        print('test cost: ', cost, 'accuracy:', accuracy)
