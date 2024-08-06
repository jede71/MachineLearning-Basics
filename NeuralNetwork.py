import numpy as np
import random

class neuralNetwork(object):
    def __init__(self, X=2, HL=[2], Y=1):   #Only one hidden layer here
        
        #Class Variables:
        self.X = X  #inputs
        self.HL = HL#Hidden Layers
        self.Y = Y  #Outputs
        
        layers = [X]+HL+[Y]       #This is the total layers, in the format of a NN as we need it, Inputs->HiddenLayers->Outputs
        
        weights = []                    #this is our array of weights
        for i in range(len(layers)-1):      #iterate going to the next layer up
            wValues = np.random.rand(layers[i], layers[i+1])    #generate random values
            weights.append(wValues)                             #append the random generated values to our weights array
        self.weights = weights
        
        derivatives = []                #derivatives array being initialised
        for i in range(len(layers)-1):  #loop through the amount of layers going next layer up
            dValues = np.zeros((layers[i], layers[i+1]))    #whereas our weights needed a random value, the derivatives will be calculated after initialisation, so we ammend the array with zeros as placeholders
            derivatives.append(dValues)                     #appending our zero values to the array
            self.derivatives = derivatives                  #set the class variable as the array
            
        outputs = []
        for i in range(len(layers)):
            outValues = np.zeros(layers[i])
            outputs.append(outValues)
            self.outputs = outputs
            
        
    #FeedForward is just our NN running forward to obtain outputs from our inputs        
        
    def feedForward(self, x):   #the 'x' here is our inputs
        output = x              #input layer's output is just the original input
        self.outputs[0] = x     #link the outputs to the class variable, will be used for back-propagation
        
        for i, w in enumerate(self.weights):    #iterate through the weights of each layer, iterate 
            nxtInput = np.dot(output, w)        #calculate the dot product of our input and weights for selected layer, this will be done once in this case for our single hidden layer
            output = self.sigmoid(nxtInput)     #this is using the sigmoid activation function for calculating the output for a certain node, very crucial part of the theory behind neural networks
            self.outputs[i+1] = output          #link to class variable for back propagation again
            
        return output           #returns the outputs of each layer
    
    
    ######### BACK PROPAGATE THE NETWORK HERE #############
    
    #Backpropagation is used for navigating the layers backwards (output end -> input end) to calculate the errors used to update the weights, so increase the weights or decrease the weights
    # depending of the derivatives of our activation function. In doing so, this is where the accuracy of the neural network is increasing and it is 'learning' how to adapt its weights so it can achieve the desired output everytime, (its important to save the weight values of NNs when you are using them to solve whatever problem you need them for, the weights are the answer in NNs)
    
    def backPropagation(self, error):       #this takes the Output error
        for i in reversed(range(len(self.derivatives))):    #'reversed' allows us to navigate through the network right to left, going backwards calculating using the derivatives
            
            output = self.outputs[i+1]  # this line is accessing the previous layer's output(as we are starting at the final output going backwards)
    
            delta = error * self.sigmoidDerivative(output)  #multiplying our error by the derivative of our activation function to achieve the d
            fixedDelta = delta.reshape(delta.shape[0], -1).T   #here we are turning the delta into an array for us to use when calculating the matrix multiplication for the derivatives
    
            currentOut = self.outputs[i]    #acccessing our current layer's output
            currentOut = currentOut.reshape(currentOut.shape[0], -1)    #reshaping our outputs into a column-array format, required format for the multiplication we are doing
    
            self.derivatives[i] = np.dot(currentOut, fixedDelta)    #This is the dot product matrix multiplication of current outputs and our delta matrix, then assigning this to our class variable
        
            error = np.dot(delta, self.weights[i].T)    #This line is essential, this calculates the error needed for the next iteration, meaning we can cycle back through the network using the necessary error from the previous layer
        ...
    
    
    #########################################################
    
    
    def trainIt(self, inputArr, target, epochs, LR):
        for i in range(epochs):                         #training cycle for as many epochs we will specify
            _errors_ = 0                                #error we will show
            
            for j, input in enumerate (inputArr):       #here we iterate through our training data 
                goal = target[j]                        #current target for the loop
                output = self.feedForward(input)        #use our feedForward function to aquire outputs for the current iteration
                
                overallError = goal - output            #this is the overall error for the network, (target minus actual output)
                
                self.backPropagation(overallError)      #Backward propagate the current epoch using its error
                self.gradientDescent(LR)
                
                _errors_ += self.msqe(goal, output)     #update the error to show to the terminal
                
            #print('epoch: ', i, "'s values")
            #print('Weights:', self.weights)
            #print('Derivatives', self.derivatives)   
            print('MSQE', np.round(_errors_, decimals=7), '\n')
        print("Here we can see the MSQE of each epoch, This is a representation of the back propagation and gradient descent functions doing their work. \n You will see that the msqe will keep getting as low as it can")
                
                
    ##########################################################

    
    def sigmoid(self,x):               #Sigmoid activation function
        y = 1.0/(1+np.exp(-x))
        return y
    
    def sigmoidDerivative(self, x):           #sigmoid function derivative
        sigD = x*(1.0-x)
        return sigD
    ...
    def msqe(self, tgt, output):                #mean square error
        msq = np.average((tgt - output)**2)
        return msq
    
    def gradientDescent(self, LR = 0.1):       #this is the gradient descent function, this is where the network will correct its error function to be as minimal as possible, 'getting the outputs back on track' given a learning rate. 
        for i in range(len(self.weights)):      #these functions are used in NN's for escaping local minimas, the hemisphere depiction where the lowest point of a hemisphere is the least possible error, this is what we want a neural network to achieve
            weight = self.weights[i]
            derivative = self.derivatives[i]
            weight += derivative * LR           #here we update the weights after applying the learning rate given
            
#####################################################################
            
if __name__ == "__main__":   #Test what we have done 

    #TRAINING DATA
    training_inputs = np.array([[random.random() for _ in range(2)] for _ in range(1000)])   #this creates a training set of inputs
    targets = np.array([[i[0] * i[1]] for i in training_inputs])                   #this creates a training set of outputs
        
    #INSTANTIATE OBJECT
    nn = neuralNetwork(2, [4], 1,)        #creates a NN with 2 inputs, 1 hidden layer and 1 ouput
    
    print("initial inputs, hidden layer and output: ", nn.X, nn.HL, nn.Y)

    #TRAIN NETWORK
    nn.trainIt(training_inputs, targets, 10, 0.1)  #trains the network with 0.1 learning rate for 10 epochs ---- Epochs is the amount of iterations through the whole network
    
    #Testing data to identify if Network trained well. 
    inputv = np.array([0.5, 0.5])      #after training this tests the train network 
    target = np.array([0.25])         # for this target value.  

    NN_output = nn.feedForward(inputv)

    print("=============== Testing the Network Screen Output ===============")
    print ("Test input is ", inputv)
    print()
    print("Target output is ",target)
    print()
    print("Neural Network actual output is ",NN_output, "there is an error (not MSQE) of ",target-NN_output)
    print("=================================================================")