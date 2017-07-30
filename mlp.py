import numpy as np

class MLP:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_out):
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_out = n_out
        self.a_hidden1 = np.zeros((n_hidden1 + 1, 1))
        self.a_hidden2 = np.zeros((n_hidden2 + 1, 1))

        sigma = 1 / (n_input + 1)
        self.w_hidden1 = np.random.rand(n_hidden1, n_input + 1) * 2 * sigma - sigma

        sigma = 1 / (n_hidden1 + 1)
        self.w_hidden2 = np.random.rand(n_hidden2, n_hidden1 + 1) * 2 * sigma - sigma

        sigma = 1 / (n_hidden2 + 1)
        self.w_out = np.random.rand(n_out, n_hidden2 + 1) * 2 * sigma - sigma

    def predict(self, inp):
        inp = np.reshape(inp, (self.n_input, 1))                            #reshape

        inp = np.insert(inp, 0, 1, axis = 0)                                #bias
        self.a_hidden1 = sigmoid(np.matmul(self.w_hidden1, inp))            #act hidden1

        self.a_hidden1 = np.insert(self.a_hidden1, 0, 1, axis = 0)          #bias
        self.a_hidden2 = sigmoid(np.matmul(self.w_hidden2, self.a_hidden1)) #act hidden2

        self.a_hidden2 = np.insert(self.a_hidden2, 0, 1, axis=0)            #bias
        out = sigmoid(np.matmul(self.w_out, self.a_hidden2))                #act out

        return out

    def train(self, lr, inps, outs):
        big_delta_out = np.zeros(np.shape(self.w_out))
        big_delta_hidden1 = np.zeros(np.shape(self.w_hidden1))
        big_delta_hidden2 = np.zeros(np.shape(self.w_hidden2))

        correct = 0

        m = np.shape(inps)[0]
        for i in range(m):
            inp = np.reshape(inps[i], (self.n_input, 1))
            a = self.predict(inp)

            y = np.reshape(outs[i], (self.n_out, 1))
            if np.argmax(a) == np.argmax(y):
                correct += 1


            delta_out = a - y

            d_act_h2 = np.multiply(self.a_hidden2, (1 - self.a_hidden2))
            delta_hidden2 = np.multiply(np.matmul(np.transpose(self.w_out), delta_out),  d_act_h2)
            delta_hidden2 = np.delete(delta_hidden2, 1, 0)

            d_act_h1 = np.multiply(self.a_hidden1, (1 - self.a_hidden1))
            delta_hidden1 = np.multiply(np.matmul(np.transpose(self.w_hidden2), delta_hidden2),  d_act_h1)
            delta_hidden1 = np.delete(delta_hidden1, 1, 0)                                                      #bias term delete

            big_delta_out = big_delta_out + np.matmul(delta_out, np.transpose(self.a_hidden2))

            big_delta_hidden2 = big_delta_hidden2 + \
                               np.matmul(delta_hidden2, np.transpose(self.a_hidden1))

            big_delta_hidden1 = big_delta_hidden1 + \
                               np.matmul(delta_hidden1, np.transpose(np.insert(inp, 0, 1, axis = 0)))

            #test
            if i%1000==0:
                print(i)


        g_out = (1 / m) * big_delta_out
        g_hidden2 = (1 / m) * big_delta_hidden2
        g_hidden1 = (1 / m) * big_delta_hidden1

        self.w_out -= lr * g_out
        self.w_hidden2 -= lr * g_hidden2
        self.w_hidden1 -= lr * g_hidden1

        print('train accuracy: ', correct / m)







def sigmoid(x):
    return 1 / (1 + np.exp(-x))
