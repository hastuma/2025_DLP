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
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr=0.1, epoch=2000, printloss_interval=100):
        #hyper par
        self.lr = lr
        self.epoch = epoch
        self.printloss_interval = printloss_interval

        # 每一層的Weight 跟bias 
        self.w1 = np.random.randn(input_size, hidden_size1)
        self.b1 = np.random.randn(1, hidden_size1)
        self.w2 = np.random.randn(hidden_size1, hidden_size2)
        self.b2 = np.random.randn(1, hidden_size2)
        self.w3 = np.random.randn(hidden_size2, output_size)
        self.b3 = np.random.randn(1, output_size)

    def forward(self, x):
        # 第一層
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        # 第二層
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)
        # 輸出層
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.output = sigmoid(self.z3)
        return self.output

    def backward(self, x, y):
        # 輸出層的 error
        output_error = self.output - y
        output_delta = output_error * derivative_sigmoid(self.output)

        # 第二層的 error
        a2_error = output_delta.dot(self.w3.T)
        a2_delta = a2_error * derivative_sigmoid(self.a2)

        # 第一層的 error
        a1_error = a2_delta.dot(self.w2.T)
        a1_delta = a1_error * derivative_sigmoid(self.a1)

        # 根據上面算出來的delta 去更新W跟bias
        self.w3 -= self.lr * self.a2.T.dot(output_delta)
        self.b3 -= self.lr * np.sum(output_delta, axis=0, keepdims=True)

        self.w2 -= self.lr * self.a1.T.dot(a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0, keepdims=True)

        self.w1 -= self.lr * x.T.dot(a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0, keepdims=True)

    def train(self, x, y):
        losses = []
        accuracies = []
        for cur_epoch in range(self.epoch):
            self.forward(x)
            loss = -np.mean(y * np.log(self.output) + (1 - y) * np.log(1 - self.output))
            predictions = (self.output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            if cur_epoch % self.printloss_interval == 0:
                print(f'Epoch {cur_epoch}/{self.epoch}, Loss: {loss}, Accuracy: {accuracy}')
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

# 設定不同的學習率
learning_rates = [0.01, 0.1, 0.5]
colors = ['r', 'g', 'b']
epochs = 8000

# 初始化資料
# x_train, y_train = generate_XOR_easy(n=100)
x_train, y_train = generate_linear(n=100)

# 訓練並記錄不同學習率的結果
all_losses = []
all_accuracies = []

for lr, color in zip(learning_rates, colors):
    nn = SimpleNeuralNetwork(input_size=2, hidden_size1=2, hidden_size2=2, output_size=1, lr=lr, epoch=epochs)
    losses, accuracies = nn.train(x_train, y_train)
    all_losses.append((losses, color, f'lr={lr}'))
    all_accuracies.append((accuracies, color, f'lr={lr}'))

# 繪製並保存訓練損失曲線
plt.figure()
for losses, color, label in all_losses:
    plt.plot(losses, color=color, label=label)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_curve.png')
plt.show()

# 繪製並保存準確率曲線
plt.figure()
for accuracies, color, label in all_accuracies:
    plt.plot(accuracies, color=color, label=label)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.show()