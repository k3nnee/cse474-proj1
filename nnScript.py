import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    '''
        n_out rows
        n_in+1 columns
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1 / (1 + np.exp(-z))


def preprocess():
    """
    Input:
     - Load the MNIST dataset from 'mnist_all.mat' file.

    Output:
     - train_data: Matrix of training set. Each row contains the feature vector of an image.
     - train_label: Vector of labels corresponding to each image in the training set.
     - validation_data: Matrix of validation set.
     - validation_label: Vector of labels corresponding to validation images.
     - test_data: Matrix of test set.
     - test_label: Vector of labels corresponding to test images.
    """

    mat = loadmat('mnist_all.mat')

    total_train_data = np.empty((0, 784))
    total_train_vector = np.array([])

    total_test_data = np.empty((0, 784))
    total_test_vector = np.array([])

    # Data loading

    for key in mat.keys():
        if key[0:5] == "train":
            total_train_data = np.concatenate((total_train_data, mat[key]))
            total_train_vector = np.append(total_train_vector, np.full(mat[key].shape[0], int(key[5])))
        elif key[0:4] == "test":
            total_test_data = np.concatenate((total_test_data, mat[key]))
            total_test_vector = np.append(total_test_vector, np.full(mat[key].shape[0], int(key[4])))

    # Feature selection

    combined_data = np.concatenate((total_train_data, total_test_data), axis=0)
    variance = np.var(combined_data, axis=0)

    total_train_data = total_train_data[:, variance > .01]
    total_test_data = total_test_data[:, variance > .01]

    # Data splitting 

    randomized_data_index = np.random.permutation(len(total_train_data))
    train_data = total_train_data[randomized_data_index[0:50000]]
    train_label = total_train_vector[randomized_data_index[0:50000]]
    validation_data = total_train_data[randomized_data_index[50000:]]
    validation_label = total_train_vector[randomized_data_index[50000:]]

    randomized_data_index = np.random.permutation(len(total_test_data))
    test_data = total_test_data[randomized_data_index]
    test_label = total_test_vector[randomized_data_index]

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, the training data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    n = training_data.shape[0] # num of training examples

    # add bias column
    ones_col = np.ones((n, 1))
    train_data_with_bias = np.concatenate([ones_col, training_data], axis = 1)


    # feedforward 

    a = np.dot(train_data_with_bias, w1.T)  
    z = sigmoid(a) 

    #add bias to hidden layer
    z_with_bias = np.concatenate([ones_col, z], axis = 1)

    b = np.dot(z_with_bias, w2.T) 
    out = sigmoid(b) 


    # error function
    # need to initialize y matrix first
    y = np.zeros((n, n_class)) #same size matrix as b and out
    y[np.arange(n), training_label.astype(int)] = 1 # apply truth values (true = 1) to the expected value of data 1...m

    # term of error for each input data - Ji(W1, W2)
    error_per_input = -(np.multiply(y, np.log(out)) + np.multiply(1-y, np.log(1-out))) # (7)

    error_func = np.sum(error_per_input) / n # (6) average error

    # backpropagation to compute gradients

    output_error = out - y  # (9) error at output layer
    
    # error at the hidden layer
    hidden_error = np.dot(output_error, w2[:, 1:]) * z * (1 - z)  # (12) backprop through hidden layer

    # gradients for w2
    grad_w2 = np.dot(output_error.T, z_with_bias) / n  # (16) output layer to hidden layer

    # gradients for w1
    grad_w1 = np.dot(hidden_error.T, train_data_with_bias) / n  # (10/11) hidden layer to input layer

    # add regularization 
    reg_cost = (lambdaval / (2 * n)) * (np.sum(np.square(w1[:, 1:])) + np.sum(np.square(w2[:, 1:])))
    obj_val = error_func + reg_cost 

    # add regularization gradients
    grad_w2[:, 1:] += (lambdaval / n) * w2[:, 1:] 
    grad_w1[:, 1:] += (lambdaval / n) * w1[:, 1:]

    # flatten gradients and concatenate
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()))

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    data_with_bias = np.hstack((np.ones((data.shape[0], 1)), data))

    input_layer_output = np.dot(data_with_bias, w1.T)   
    input_layer_output_after_sigmoid = sigmoid(input_layer_output)  

    input_with_bias = np.hstack((np.ones((input_layer_output_after_sigmoid.shape[0], 1)), input_layer_output_after_sigmoid))

    hidden_layer_output = np.dot(input_with_bias, w2.T)
    hidden_layer_output_after_sigmoid = sigmoid(hidden_layer_output)  

    for arr in hidden_layer_output_after_sigmoid:
        labels = np.append(labels, np.argmax(arr))  

    return labels


"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 176

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


