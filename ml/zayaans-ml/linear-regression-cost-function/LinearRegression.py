import numpy as np
import matplotlib.pyplot as plt
import math
import time

class LinearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y # this is the result (not prediction)
        self.b = [0, 0] # set w and b = 0 (y = wx + b)

    def predict(self, X=[]):
        Y_prediction = np.array([]) # creating array for y-hat buffer
        if not X: X = self.X # if X is not initialized
        b = self.b
        for x in X: 
            Y_prediction = np.append(Y_prediction, b[0]) + (b[1] * x)
        return Y_prediction # return y-hat buffer
    
    def update_coefficients(self, learning_rate):
        Y_prediction = self.predict()
        Y = self.Y
        m = len(Y) # number of training samples
        # now we update b and w using 
        # for b
        self.b[0] = self.b[0] - (learning_rate * ((1 / m) * np.sum(Y_prediction - Y)))
        self.b[1] = self.b[1] - (learning_rate * ((1 / m) * np.sum((Y_prediction - Y) * self.X)))

    def get__current_accuracy(self, Y_prediction):
        prediction, actual = Y_prediction, self.Y
        n = len(Y_prediction) # number of predictions

        for i in range(n):
            if actual[i] != 0:
                return abs(1- np.sum(abs(prediction[i] - actual[i]) / actual[i]) / n)
    
    def compute_loss(self, Y_prediction):
        m = len(self.Y)
        J = (1 / 2 * m) * (np.sum((Y_prediction - self.Y) ** 2))
        return J
    
    def plot_best_fit(self, Y_prediction, figure):
        figureObject = plt.figure(figure)
        # create a scatter plot
        plt.scatter(self.X, self.Y, color="b")
        # add the line in
        plt.plot(self.X, Y_prediction, color="g")
        figureObject.show()
    
if __name__ == "__main__":
    # create X and Y
    X = np.array([i for i in range(11)])
    Y = np.array([2*i for i in range(11)])
    
    regressionModel = LinearRegression(X, Y)

    iterations = 0 
    steps = 100
    learning_rate = 0.01
    costs_buffer = []

    # plot hte original best-fit line
    Y_prediction = regressionModel.predict()
    regressionModel.plot_best_fit(Y_prediction, "Initial Line of Best Fit")

    while 1:
        Y_prediction = regressionModel.predict()
        # generate the cost on y-hat
        cost = regressionModel.compute_loss(Y_prediction)
        # add that cost to the buffer
        costs_buffer.append(cost)
        regressionModel.update_coefficients(learning_rate)

        iterations += 1
        # get the epochs (cycles of iterations per step)
        if iterations % steps == 0: 
            print(iterations, "epochs elapsed")
            print("Current accuracy is: ", regressionModel.get__current_accuracy(Y_prediction))

            end = input("End run? y/n")
            if end == "y":
                break
    
    regressionModel.plot_best_fit(Y_prediction, "Final Line of Best Fit")
    time.sleep(5)

    # Create a plot to see if the cost/loss function decreases
    plot = plt.figure("Verification")
    plt.plot(range(iterations), costs_buffer, color="b")
    plot.show()

    # if the user wants to predict using the model
    regressionModel.predict([i for i in range(10)])



