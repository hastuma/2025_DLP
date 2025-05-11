import numpy as np 
import matplotlib.pyplot as plt
import pickle

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy(n=100):
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, lr=0.1, epoch=2000, printloss_interval=100):
        #hyper par
        self.lr = lr
        self.epoch = epoch
        self.printloss_interval = printloss_interval

        # 每一層的Weight 跟bias 
        self.weights = []
        self.biases = []

        # 初始化第一層
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.random.randn(1, hidden_sizes[0]))

        # 初始化隱藏層
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]))
            self.biases.append(np.random.randn(1, hidden_sizes[i]))

        # 初始化輸出層
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.random.randn(1, output_size))

    def forward(self, x):
        self.activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            a = sigmoid(z)
            self.activations.append(a)
        self.output = self.activations[-1]
        return self.output

    def backward(self, x, y):
        deltas = [self.output - y]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = deltas[-1].dot(self.weights[i].T) * derivative_sigmoid(self.activations[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.activations[i].T.dot(deltas[i])
            self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, x, y):
        losses = []
        accuracies = []
        for cur_epoch in range(self.epoch):
            self.forward(x)
            loss = -np.mean(y * np.log(self.output) + (1 - y) * np.log(1 - self.output))
            predictions = (self.output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            # if cur_epoch % self.printloss_interval == 0:
            #     print(f'Epoch {cur_epoch}/{self.epoch}, Loss: {loss}, Accuracy: {accuracy}')
            losses.append(loss)
            accuracies.append(accuracy)
            self.backward(x, y)
        
        return losses, accuracies

    def save_model(self, filename='model.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename='model.pkl'):
        with open(filename, 'rb') as file:
            return pickle.load(file)

# 設定不同的隱藏層數量
hidden_layer_configs = [[2], [5], [10]]
colors = ['r', 'g', 'b']

# 初始化資料
x_train_linear, y_train_linear = generate_linear(n=100)
x_train_xor, y_train_xor = generate_XOR_easy(n=100)

# 訓練並記錄不同隱藏層數量的結果
all_losses_linear = []
all_accuracies_linear = []
all_losses_xor = []
all_accuracies_xor = []

# Linear task
for hidden_sizes, color in zip(hidden_layer_configs, colors):
    print(hidden_sizes)
    nn = SimpleNeuralNetwork(input_size=2, hidden_sizes=hidden_sizes, output_size=1, lr=0.1, epoch=5000)
    losses, accuracies = nn.train(x_train_linear, y_train_linear)
    all_losses_linear.append((losses, color, f'hidden_layers={hidden_sizes}'))
    all_accuracies_linear.append((accuracies, color, f'hidden_layers={hidden_sizes}'))

# XOR task
for hidden_sizes, color in zip(hidden_layer_configs, colors):
    nn = SimpleNeuralNetwork(input_size=2, hidden_sizes=hidden_sizes, output_size=1, lr=0.2, epoch=10000)
    losses, accuracies = nn.train(x_train_xor, y_train_xor)
    all_losses_xor.append((losses, color, f'hidden_layers={hidden_sizes}'))
    all_accuracies_xor.append((accuracies, color, f'hidden_layers={hidden_sizes}'))

# 繪製並保存訓練損失曲線 (Linear)
plt.figure()
for losses, color, label in all_losses_linear:
    plt.plot(losses, color=color, label=label, linewidth=1)
plt.title('Training Loss (Linear)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_curve_linear.png')
plt.show()

# 繪製並保存準確率曲線 (Linear)
plt.figure()
for accuracies, color, label in all_accuracies_linear:
    plt.plot(accuracies, color=color, label=label, linewidth=1)
plt.title('Accuracy (Linear)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve_linear.png')
plt.show()

# 繪製並保存訓練損失曲線 (XOR)
plt.figure()
for losses, color, label in all_losses_xor:
    plt.plot(losses, color=color, label=label, linewidth=1)
plt.title('Training Loss (XOR)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_curve_xor.png')
plt.show()

# 繪製並保存準確率曲線 (XOR)
plt.figure()
for accuracies, color, label in all_accuracies_xor:
    plt.plot(accuracies, color=color, label=label, linewidth=1)
plt.title('Accuracy (XOR)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve_xor.png')
plt.show()