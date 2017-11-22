import numpy as np 
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
# plt.scatter(X, Y)
# plt.show()

X_train, Y_train = X[:160], Y[:160]     # train 前 160 data points
X_test, Y_test = X[160:], Y[160:]       # test 后 40 data points

#build model
model= Sequential()
model.add(Dense(output_dim=1,input_dim=1))
#choose loss function
model.compile(loss='mse',optimizer='sgd')

#train
print("training__________________________________")
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    if step %100==0:
        #maybe str()
        print('traincost',cost)


#test
print("test__________________________________")
cost =model.evaluate(X_test,Y_test,batch_size=40)
W,b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
