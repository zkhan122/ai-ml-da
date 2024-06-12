import numpy as np
import matplotlib.pyplot as plt
import math
import time


class LinearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.wb = [0, 0] # w, b

    def predict(self, X=[]):
        Y_prediction = np.array([])
        if len(X) == 0: X = self.X # initialize X if not created
        X = np.array(X)
        wb = self.wb
        for x in X:
            Y_prediction = np.append(Y_prediction, wb[1]) + (wb[0] * x)
        return Y_prediction # this is returning the y-hat buffer


    def update_coefficients(self, learning_rate):
        Y_prediction = self.predict(self.X)
        Y = self.Y
        m = len(Y) # number of training samplesd
        # for w
        #self.wb[0] = self.wb[0] - (learning_rate * ((1 / m) * np.sum((Y_prediction - Y) * self.X)))
        # for b
        #self.wb[1] = self.wb[1] - (learning_rate * ((1 / m) * np.sum((Y_prediction - Y))))
        
        # Update for w
        dw = (1 / m) * np.dot(X.T, (Y_prediction - Y))
        self.wb[0] = self.wb[0] - learning_rate * dw

        # Update for b
        db = (1 / m) * np.sum(Y_prediction - Y)
        self.wb[1] = self.wb[1] - learning_rate * db


    def get_current_accuracy(self, Y_prediction):
        prediction, actual = Y_prediction, self.Y
        ss_res = np.sum((actual - prediction) ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared

    def compute_loss(self, Y_prediction):
        m = len(self.Y)
        J_cost_function = (1 / 2 * m) * (np.sum((Y_prediction - self.Y) ** 2))
        return J_cost_function

    def plot_best_fit(self, Y_prediction, figure):
        figureObject = plt.figure(figure)
        # create a scatter plot
        plt.scatter(self.X, self.Y, color="b")
        # add the line in
        plt.plot(self.X, Y_prediction, color="g")
        figureObject.show()

if __name__== "__main__":
    X = np.array([i for i in range(11)])
    Y = np.array([2*i for i in range(11)])

    print(Y)
    
    regressionModel = LinearRegression(X, Y)
    
    print(regressionModel)

    iterations = 0
    steps = 100
    learning_rate = 0.0000001
    costs_buffer = []

    Y_prediction = regressionModel.predict()
    regressionModel.plot_best_fit(Y_prediction, "Initial Line of Best Fit")

    while 1:
        Y_prediction = regressionModel.predict()
        # here we generate the cost on y-hat
        cost = regressionModel.compute_loss(Y_prediction)
        # add cost to buffer
        costs_buffer.append(cost)
        regressionModel.update_coefficients(learning_rate)

        iterations += 1
        if iterations % steps == 0:
            print("\n", iterations, "epochs elapsed")
            print("Current accuracy is: ", regressionModel.get_current_accuracy(Y_prediction))

            end = input("End run? y/n: ")
            if end == "y":
                break

    regressionModel.plot_best_fit(Y_prediction, "Final Line of Best Fit")
    # time.sleep(10)

    # creating a plot to see if the cost/loss function decreases
    plot = plt.figure("Verification")
    plt.plot(range(iterations), costs_buffer, color="b")
    time.sleep(10)

    print(regressionModel.predict([i for i in range(10)]))
