import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_dim=2, hidden_dim1=4, hidden_dim2=4, output_dim=1, lr=0.1, epoch=10000, use_activation=True):
        self.w1 = np.random.randn(input_dim, hidden_dim1) 
        self.b1 = np.zeros((1, hidden_dim1))
        self.w2 = np.random.randn(hidden_dim1, hidden_dim2) 
        self.b2 = np.zeros((1, hidden_dim2))
        self.w3 = np.random.randn(hidden_dim2, output_dim) 
        self.b3 = np.zeros((1, output_dim))
        self.lr = lr
        self.epoch = epoch
        self.printloss_interval = 1000
        self.use_activation = use_activation
        if self.use_activation:
            self.activation = sigmoid
            self.derivative_activation = derivative_sigmoid
        else:
            self.activation = lambda x: x  # 線性激活
            self.derivative_activation = lambda x: 1  # 線性激活的導數

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.activation(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.output = sigmoid(self.z3)  # 輸出層始終使用 sigmoid
        return self.output

    def backward(self, x, y):
        output_error = self.output - y
        output_delta = output_error * derivative_sigmoid(self.output)
        a2_error = output_delta.dot(self.w3.T)
        a2_delta = a2_error * self.derivative_activation(self.a2)
        a1_error = a2_delta.dot(self.w2.T)
        a1_delta = a1_error * self.derivative_activation(self.a1)
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
            self.output = np.clip(self.output, 1e-10, 1 - 1e-10)
            loss = -np.mean(y * np.log(self.output) + (1 - y) * np.log(1 - self.output))
            predictions = (self.output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            if cur_epoch % self.printloss_interval == 0:
                print(f'Epoch {cur_epoch}/{self.epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
            losses.append(loss)
            accuracies.append(accuracy)
            self.backward(x, y)
        return losses, accuracies

    def predict(self, x):
        pred_y = self.forward(x)
        return (pred_y > 0.5).astype(int)

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
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
    accuracy = np.mean(y == pred_y) * 100
    print(f'Accuracy: {accuracy:.1f}%')


def plot_accuracies(acc_with_act, acc_without_act, dataset_name):
    plt.figure(figsize=(10, 5))
    plt.plot(acc_with_act, label='Without Activation', color='blue')
    plt.plot(acc_without_act, label='With Activation', color='red')
    plt.title(f'Accuracy Comparison for {dataset_name} Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dataset_name.lower()}_accuracy_comparison.png')
    plt.show()



x_linear, y_linear = generate_linear(n=100)
x_xor, y_xor = generate_XOR_easy()

# 1. linear + 有 sigmoid
print("\nTraining Linear with Sigmoid:")
nn_linear_act = NeuralNetwork(use_activation=True)
losses_linear_act, accuracies_linear_act = nn_linear_act.train(x_linear, y_linear)
pred_y_linear_act = nn_linear_act.predict(x_linear)
show_result(x_linear, y_linear, pred_y_linear_act, filename='linear_with_sigmoid.png')

# 2. linear + 無 sigmoid 
print("\nTraining Linear without Sigmoid:")
nn_linear_no_act = NeuralNetwork(use_activation=False)
losses_linear_no_act, accuracies_linear_no_act = nn_linear_no_act.train(x_linear, y_linear)
pred_y_linear_no_act = nn_linear_no_act.predict(x_linear)
show_result(x_linear, y_linear, pred_y_linear_no_act, filename='linear_without_sigmoid.png')

# 3. XOR  + 有 sigmoid
print("\nTraining XOR with Sigmoid:")
nn_xor_act = NeuralNetwork(use_activation=True, lr=0.2, epoch=10000)
losses_xor_act, accuracies_xor_act = nn_xor_act.train(x_xor, y_xor)
pred_y_xor_act = nn_xor_act.predict(x_xor)
show_result(x_xor, y_xor, pred_y_xor_act, filename='xor_with_sigmoid.png')

# 4. XOR  + 無 sigmoid (隱藏層)
print("\nTraining XOR without Sigmoid:")
nn_xor_no_act = NeuralNetwork(use_activation=False)
losses_xor_no_act, accuracies_xor_no_act = nn_xor_no_act.train(x_xor, y_xor)
pred_y_xor_no_act = nn_xor_no_act.predict(x_xor)
show_result(x_xor, y_xor, pred_y_xor_no_act, filename='xor_without_sigmoid.png')


plot_accuracies(accuracies_linear_act, accuracies_linear_no_act, 'Linear')
plot_accuracies(accuracies_xor_act, accuracies_xor_no_act, 'XOR')

