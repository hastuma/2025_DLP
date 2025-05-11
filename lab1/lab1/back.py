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

def show_result(x, y, pred_y, filename='result.png'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  
    
    for ax, title, labels in zip(axes, ['Ground truth', 'Predict result'], [y, pred_y]):
        ax.set_title(title, fontsize=18)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal') 
        
        for i in range(x.shape[0]):
            color = 'ro' if labels[i] == 0 else 'bo'
            ax.plot(x[i][0], x[i][1], color)

    plt.savefig(filename)
    plt.show()

    # Print Ground Truth and Prediction for each iteration
    for i in range(x.shape[0]):
        print(f'Iter: {i} | Ground Truth: {y[i][0]} | Prediction: {pred_y[i][0]}')

    # Calculate and print accuracy
    accuracy = np.mean(y == pred_y) * 100
    print(f'Accuracy: {accuracy:.1f}%')

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
        for cur_epoch in range(self.epoch):
            self.forward(x)
            loss = -np.mean(y * np.log(self.output) + (1 - y) * np.log(1 - self.output))
            predictions = (self.output > 0.5).astype(int)
            if cur_epoch % self.printloss_interval == 0:
                print(f'Epoch {cur_epoch}/{self.epoch}, Loss: {loss}')
            losses.append(loss)
            self.backward(x, y)
        
        # 繪製並保存訓練損失曲線
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # 添加虛線
        y_ticks = plt.gca().get_yticks()
        for y_tick in y_ticks:
            plt.axhline(y=y_tick, color='gray', linestyle='--', linewidth=0.5)
        
        plt.savefig('loss_curve.png')
        plt.show()
        
        return losses

    def save_model(self, filename='model.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename='linearmodel.pkl'):
        with open(filename, 'rb') as file:
            return pickle.load(file)

# 有兩種 data的方式，一種是linear，另一種是XOR_easy
# data_type = 'XOR_easy'
data_type ='xor'
lr = 0.1
load_model = False

if data_type == 'linear':
    x_train, y_train = generate_linear(n=100)
    x_test, y_test = generate_linear(n=100)
    epoch = 5000
else:
    x_train, y_train = generate_XOR_easy(n=100)
    x_test, y_test = generate_XOR_easy(n=100)
    epoch = 10000
    lr = 0.2

if load_model:
    nn = SimpleNeuralNetwork.load_model()
else:
    # 初始化並訓練神經網路
    nn = SimpleNeuralNetwork(input_size=2, hidden_size1=2, hidden_size2=2, output_size=1, lr=lr, epoch=epoch)
    losses = nn.train(x_train, y_train)
    nn.save_model()

# 顯示測試結果並保存圖片
pred_y_test = nn.forward(x_test)
show_result(x_test, y_test, (pred_y_test > 0.5).astype(int), filename='test_result.png')