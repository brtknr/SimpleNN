import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from sklearn.cross_validation import train_test_split

class Sigmoid(object):
    f = lambda x: 1/(1+np.exp(-x))
    df = lambda x: x*(1- x)

class Tanh(object):
    f = lambda x: np.tanh(x)
    df = lambda x: 1 - x**2

class ReLU(object):
    f = lambda x: np.abs(x*(x>0))
    df = lambda x: np.float_(x>0)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def softmax(x):
    e_x = np.exp(x-x.max(axis=1).reshape(-1,1)) 
    return e_x/e_x.sum(axis=1).reshape(-1,1)

def cross_entropy(y,yp):
    return (y*np.log(yp)+(1-y)*np.log(1-yp)).mean()

def pause():
    for _ in range(100000000):
        pass

class SimpleNN(object):

    def __init__(self, num_nodes):
        # Initialise synapses
        self.syn = list()
        for i in range(len(num_nodes)-1):
            self.syn.append(2*np.random.random((num_nodes[i],num_nodes[i+1]))-1)

    def dropout(self,X,keep_prob):
        return (np.random.random(X.shape)<keep_prob)*X

    def fore_prop(self,X,keep_prob):
        # Foreward propagation    
        self.layers = list()
        this_layer = X
        for i,syn in enumerate(self.syn):
            this_layer = self.activation.f(this_layer.dot(syn))
            this_layer = self.dropout(this_layer,keep_prob)
            self.layers.append(this_layer)
        self.layers[-1] = softmax(self.layers[-1])

    def back_prop(self,X):
        # Back propagation
        this_delta = self.activation.df(self.layers[-1]) * self.loss
        for this_layer, this_syn in reversed(list(zip(self.layers[:-1], self.syn[1:]))):
            # print(this_delta)
            this_syn += self.alpha*this_layer.T.dot(this_delta)
            this_delta = this_delta.dot(this_syn.T) * self.activation.df(this_layer)
        # print(this_delta)            
        self.syn[0] += self.alpha*X.T.dot(this_delta)

    def learn(self,train_X, test_X, train_y, test_y, alpha=0.001, keep_prob=1.0,activation=Tanh,epochs=1000,batch_size=10):
        self.activation = activation
        self.alpha = alpha

        CORRECT_COLOUR = 42
        INCORRECT_COLOUR = 41

        # self.grad = deque(maxlen=1)
        accuracy = list()
        for epoch in range(epochs):
            for i in range(len(train_X)//batch_size):
                slide = i*batch_size
                batch_X = train_X[slide:slide+batch_size]
                batch_y = train_y[slide:slide+batch_size]
                self.fore_prop(batch_X,keep_prob)
                # print (self.layers[-1])
                self.loss = batch_y - self.layers[-1]
                # print (loss)
                self.back_prop(batch_X)
                # pause()

                if np.isnan(snn.layers[-1]).any():
                    print (self.prev_y)
                    print (self.prev_layer)
                    print (self.prev_y-self.prev_layer)
                    raise ValueError('NaNs found')
                self.prev_y = batch_y.copy()
                self.prev_layer = self.layers[-1].copy()

            prediction = self.predict(test_X)
            success = np.argmax(test_y, axis=1) == prediction

            test_accuracy = np.mean(success)
            accuracy.append(test_accuracy)

            if epoch%10 == 0 or test_accuracy == 1:
                print("Epoch = {}, test accuracy = {:0.2f}%".format(epoch + 1, 100. * test_accuracy))

                output = ''
                for p,s in zip(prediction,success):
                    # Guide on how to colour text:
                    # http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
                    output += '\x1b[5;37;{}m {} \x1b[0m'.format(CORRECT_COLOUR if s else INCORRECT_COLOUR, p)
                print (output)

                if test_accuracy == 1:
                    break

        return accuracy

    def predict(self,X,keep_prob=1.0):
        self.fore_prop(X,keep_prob=keep_prob)
        return self.layers[-1].argmax(axis=1)

def get_modulus_data():
    """ Generate a dataset where the output is the modulus of the sum of a random array """
    all_X = np.rint(np.random.random((1500,6)))
    target = np.int_(all_X.sum(axis=1)%3)
    # target = np.int_(np.round(np.random.uniform(0,2,150)))

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!

    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

train_X, test_X, train_y, test_y = get_modulus_data()


plt.figure()
plt.ion()
plt.show()

for hl in range(11,21):
    num_nodes = [train_X.shape[1],hl,hl,hl,hl,train_y.shape[1]]
    snn = SimpleNN(num_nodes)
    accuracy = snn.learn(train_X, test_X, train_y, test_y, epochs=500, batch_size=5)
    plt.plot(1-np.square(accuracy),label='# hl = {}'.format(hl))
    plt.draw()
    plt.show()

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.savefig('{}.pdf'.format('-'.join(str(n) for n in num_nodes)))
