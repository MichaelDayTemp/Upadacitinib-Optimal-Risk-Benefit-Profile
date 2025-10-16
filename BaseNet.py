import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

#Test labled Data Set
def RA_Data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


# Class for Creating Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = 0.01 * np.random.randn(1, n_neurons)
        self.inputs = n_inputs
        self.neurons = n_neurons
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#Class for calculating Rectified Linear
class Activation_ReLU:
    def forward(self, inputs):
        self.output = (np.maximum(0, inputs))

#Class for Softmax (Probability distribution)
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#Class for calculating Loss (Cross Catagorical Entropy) for multiclass dataset
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return(data_loss)
class Loss_CCE(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negloglikelihoods = -np.log(correct_confidences)
        return negloglikelihoods #probability distribution

#Input and Target Values
X, y = RA_Data(100, 3)
print(X)
"""
#Displays inital data
plt.scatter(X[:,0], X[:,1])
plt.show()
plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()"""

#Initalizing Variables & Classes
lowest_loss = 999999
num_examples = X.shape[0]
loss_function = Loss_CCE()
step_size = 1e-0
reg = 1e-3

#Forward Pass
dense1 = Layer_Dense(2,100)
Activation1 = Activation_ReLU()
dense2 = Layer_Dense(100,3)
Activation2 = Activation_Softmax()
dense1.forward(X)
Activation1.forward(dense1.output)
dense2.forward(Activation1.output)
Activation2.forward(dense2.output)

#Looping Forward passes w/ backpropagation (Optimization)
for iteration in range(65000):
    #Rectified Linear Adjustments
    dense1.forward(X)
    Activation1.forward(dense1.output)
    
    #Prob Distrubution
    dense2.forward(Activation1.output)
    Activation2.forward(dense2.output)
    
    #Calculating Loss
    loss = loss_function.calculate(Activation2.output, y)
    predictions = np.argmax(Activation2.output, axis=1)
    accuracy = np.mean(predictions==y)
    
    #Status Updates
    if iteration % 1000 == 0:
        print(f'at iteration {iteration}: loss  {loss}, accuracy: {accuracy}')
    if loss <= lowest_loss:
        lowest_loss = loss
        best_accuracy = accuracy
        best_iteration = iteration
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
    
    #Compute Gradient
    dscores = Activation2.output
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
        
    dW2 = np.dot(Activation1.output.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # backprop into hidden layer
    dhidden = np.dot(dscores, dense2.weights.T)
    # backprop the Rectified Linear
    dhidden[Activation1.output <= 0] = 0
    # Finds the instantaneous rate of change
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True) 

    # Update paramaters according to ROC
    dense1.weights += -(step_size * dW )
    dense1.biases += -(step_size * db)
    dense2.weights += -(step_size * dW2)
    dense2.biases += -(step_size * db2)

# evaluate training set accuracy
Activation1.output = np.maximum(0, np.dot(X, dense1.weights) + dense1.biases)
scores = np.dot(Activation1.output, dense2.weights) + dense2.biases
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

print("Lowest Loss achieved was: " + str(lowest_loss) + " at iteration " + str(best_iteration))


# plots the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], dense1.weights) + dense1.biases), dense2.weights) + dense2.biases
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()