import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import json

class NNet(object):
    def __init__(self, input_size, output_size, layers, activation_fn='swish', lr=0.001):
        self.ip_size = input_size
        self.op_size = output_size
        self.layers = layers
        self.n_layers = len(layers)
        self.hidden = [0]*(self.n_layers + 2)
        self.acts = [0]*(self.n_layers + 2)
        self.hidden_grads = [0]*(self.n_layers + 2)
        self.weight_grads = [0]*(self.n_layers + 1)
        self.bias_grads = [0]*(self.n_layers + 1)
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-6
        self.steps = 0
        self.global_step = 0
        self.weights_init(self.ip_size, self.op_size, self.layers)
        self.bias_init(self.ip_size, self.op_size, self.layers)
        self.act_fn = activation_fn
    
    def xavier_init(self, ip_size, op_size):
        stddev = np.sqrt(6/(ip_size + op_size))
        return np.random.uniform(low=-stddev, high=stddev, size=(ip_size, op_size))
    
    def weights_init(self, ip_size, op_size, layers):
        self.weights = []
        self.weights1 = []
        self.weights2 = []
        for (i, j) in zip([ip_size] + layers, layers + [op_size]):
            self.weights.append(self.xavier_init(i, j))
            self.weights1.append(np.zeros(shape=(i, j)))
            self.weights2.append(np.zeros(shape=(i, j)))
    
    def bias_init(self, ip_size, op_size, layers):
        self.biases = []
        self.bias1 = []
        self.bias2 = []
        for i in (layers + [op_size]):
            self.biases.append(np.zeros(shape=(i,)))
            self.bias1.append(np.zeros(shape=(i,)))
            self.bias2.append(np.zeros(shape=(i,)))

    def softmax(self, activations):
        # Numerically stable activations.
        activations = activations - np.max(activations, axis=1, keepdims=True)
        eacts = np.exp(activations)
        if activations.ndim == 1:
            return eacts / np.sum(eacts)
        elif activations.ndim == 2:
            return eacts / np.sum(eacts, axis=1, keepdims=True)

    def grad_softmax(self, preds, y):
        return y - preds
    
    def activation(self, acts):
        if self.act_fn =='relu':
            acts[acts < 0]=0
            return acts
        elif self.act_fn == 'sigmoid':
            return (1 / (1 + np.exp(-acts)))
        elif self.act_fn == 'swish':
            return (1 / (1 + np.exp(-acts))) * acts
        elif self.act_fn == 'tanh':
            return np.tanh(acts)
    
    def grad_activation(self, acts_post, acts_pre):
        if self.act_fn == 'relu':
            grads = np.zeros(shape=acts_post.shape)
            grads[acts_post > 0] = 1
            return grads
        elif self.act_fn == 'sigmoid':
            return acts_post * (1 - acts_post)
        elif self.act_fn == 'swish':
            return acts_post + (1 / (1 + np.exp(-acts_pre))) * (1 - acts_post)
        elif self.act_fn == 'tanh':
            return 1 - np.power(acts_post, 2)
            

    def forward(self, x):
        self.hidden[0] = x
        for i in range(1, len(self.hidden)):
            self.acts[i] = np.dot(self.hidden[i-1], self.weights[i-1]) + self.biases[i-1]
            if i == (len(self.hidden) - 1):
                self.hidden[i] = self.acts[i]
            self.hidden[i] = self.activation(self.acts[i])

        self.hidden[-1] = self.softmax(self.hidden[-1])
    
    def loss(self, x, preds, y):
        return -1 * np.sum(y * np.log(preds)) / preds.shape[0]
    
    def backward(self, x, y):
        preds = self.hidden[-1]
        op_grad = self.grad_softmax(y, preds)
        self.hidden_grads = [0]*len(self.hidden)
        self.hidden_grads[-1] = op_grad

        for i in range(self.n_layers + 1, 0, -1):
            self.hidden_grads[i-1] = np.dot(self.hidden_grads[i], self.weights[i-1].T) * self.grad_activation(self.hidden[i-1], self.acts[i-1])
        
        for i in range(self.n_layers, -1, -1):
            self.weight_grads[i] = np.dot(self.hidden[i].T, self.hidden_grads[i+1]) / self.hidden[i].shape[0]
        
        for i in range(self.n_layers, -1 , -1):
            self.bias_grads[i] = np.sum(self.hidden_grads[i+1], axis=0) / self.hidden_grads[i+1].shape[0]
    
    def apply_grads(self):
        # Use ADAM Optimizer.
        for i in range(len(self.weights)):
            # Momentum updates for weights.
            self.weights1[i] = self.beta1 * self.weights1[i]  + (1 - self.beta1) * self.weight_grads[i]
            self.weights2[i] = self.beta2 * self.weights2[i]  + (1 - self.beta2) * np.power(self.weight_grads[i], 2)
            
            # Momentum updates for biases.
            self.bias1[i] = self.beta1 * self.bias1[i]  + (1 - self.beta1) * self.bias_grads[i]
            self.bias2[i] = self.beta2 * self.bias2[i]  + (1 - self.beta2) * np.power(self.bias_grads[i], 2)
                
            # Bias correction to make estimates unbiased. # The actual update step.
            t1 = self.weights1[i] / (1 - (self.beta1**self.steps))
            t2 = self.weights2[i] / (1 - (self.beta2**self.steps))
            self.weights[i] = self.weights[i] - self.lr * (t1 / (np.sqrt(t2) + self.eps))
            #self.weights[i] = self.weights[i] - self.lr * self.weight_grads[i]
            
            t1 = self.bias1[i] / (1 - (self.beta1**self.steps))
            t2 = self.bias2[i] / (1 - (self.beta2**self.steps))
            self.biases[i] = self.biases[i] - self.lr * (t1 / (np.sqrt(t2) + self.eps))
            #self.biases[i] = self.biases[i] - self.lr * self.bias_grads[i]
    
    def single_pass(self, x, y):
        self.forward(x)
        l = self.loss(x, self.hidden[-1], y)
        self.backward(x, y)
        self.apply_grads()
        return l
    
    def train(self, x, y, epochs):
        l = 0
        for i in range(epochs):
            self.steps += 1
            l += self.single_pass(x, y)
            #print("Epoch: %d, loss: %.3f" % (self.steps, l))
        return l / epochs
    
    def minibatch_train(self, batch_gen, max_iter):
        cnt = 0
        l = 0
        score_log = []
        self.global_step = 0
        for (x, y, one_pass) in batch_gen:
            l += self.train(x, y, 1)
            cnt += 1
            if one_pass:
                train_scores = self.score(x, y)
                epoch_scores = [self.global_step, float(l)/cnt, train_scores]
                
                if batch_gen.splitted:
                    val_scores = self.score(*batch_gen.get_validation_data())
                    epoch_scores.append(val_scores)
                
                score_log.append(tuple(epoch_scores))
                self.print_epoch_scores(epoch_scores)
                
                cnt = 0
                l = 0
                self.global_step += 1
                if self.global_step > max_iter:
                    break

        return score_log
    
    def print_epoch_scores(self, epoch_scores):
        s = "Epoch %d, " % (epoch_scores[0])
        s += "Loss %.3f, " % (epoch_scores[1])
        s += "train_accuracy: %.3f, " % (epoch_scores[2]['accuracy'])
        if len(epoch_scores) >= 4:
            s += "valid_accuracy: %.3f" % (epoch_scores[3]['accuracy'])
        print(s)

    def predict(self, x):
        self.forward(x)
        return self.hidden[-1]
    
    def score(self, x, y):
        y_hat = self.predict(x).argmax(axis=1)
        y_true = y.argmax(axis=1)
        return {
            'f1-micro': f1_score(y_true, y_hat, average='micro'),
            'f1-macro': f1_score(y_true, y_hat, average='macro'),
            'accuracy': accuracy_score(y_true, y_hat)
        }
    
    def store(self, fname):
        config = {
            'ip_size': self.ip_size,
            'hidden_layers': self.layers,
            'op_size': self.op_size,
            'weights': [w.tolist() for w in nn.weights],
            'biases': [b.tolist() for b in nn.biases],
            'activation': self.act_fn
        }
        f = open('%s.json' % fname, 'w')
        f.write(json.dumps(config))
        f.close()
    
    @classmethod
    def load(cls, fname):
        f = open('%s.json' % fname, 'r')
        data = json.loads(f.read())
        f.close()
        nn = cls(data['ip_size'], data['op_size'], data['hidden_layers'], activation_fn=data['activation'])
        nn.weights = data['weights']
        nn.biases = data['biases']
        return nn
