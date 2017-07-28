import numpy as np

class MLP:
    def __init__(self, n_input, n_hidden, n_out):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.a_hidden = np.zeros((n_hidden + 1, 1))

        sigma = 1 / (n_input + 1)
        self.w_hidden = np.random.rand(n_hidden, n_input + 1) * 2 * sigma - sigma

        sigma = 1 / (n_hidden + 1)
        self.w_out = np.random.rand(n_out, n_hidden + 1) * 2 * sigma - sigma

    def predict(self, inp):
        inp = np.reshape(inp, (self.n_input, 1))
        inp = np.insert(inp, 0, 1, axis = 0)

        self.a_hidden = sigmoid(np.matmul(self.w_hidden, inp))
        self.a_hidden = np.insert(self.a_hidden, 0, 1, axis = 0)

        out = sigmoid(np.matmul(self.w_out, self.a_hidden))

        return out

    def train(self, lr, inps, outs):
        big_delta_out = np.zeros(np.shape(self.w_out))
        big_delta_hidden = np.zeros(np.shape(self.w_hidden))

        m = np.shape(inps)[0]
        for i in range(m):
            inp = np.reshape(inps[i], (self.n_input, 1))
            a = self.predict(inp)
            delta_out = a - np.reshape(outs[i], (self.n_out, 1))

            d_act = np.multiply(self.a_hidden, (1 - self.a_hidden))
            delta_hidden = np.multiply(np.matmul(np.transpose(self.w_out), delta_out),  d_act)

            big_delta_out = big_delta_out + np.matmul(delta_out, np.transpose(self.a_hidden))
            big_delta_hidden = big_delta_hidden + \
                               np.matmul(delta_hidden, np.transpose(np.insert(inp, 0, 1, axis = 0)))

        g_out = (1 / m) * big_delta_out
        g_hidden = (1 / m) * big_delta_hidden

        self.w_out -= lr * g_out
        self.w_hidden -= lr * g_hidden







def sigmoid(x):
    return 1 / (1 + np.exp(-x))
