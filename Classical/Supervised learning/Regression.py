import numpy as np
import pandas as pd


data = pd.read_csv("data/Admission_Predict.csv")
data.drop("Serial No.", axis="columns", inplace=True)
print(data.columns.values)


class Regression:
    def __init__(self, type: str):
        self.parameters = {}
        self.type = type

    def predict(self, x):
        m = self.parameters["m"]
        c = self.parameters["c"]
        return x * m + c

    def _cost(self, pred, y):
        return np.mean((y - pred) ** 2)

    def _grad(self, pred, y, x):
        derivatives = {}
        dc = 2 * np.mean((pred - y))
        dm = 2 * np.mean((pred - y) * x)
        derivatives["dm"] = dm
        derivatives["dc"] = dc
        return derivatives

    def _back_prop(self, pred, y, x, lr):
        grad = self._grad(pred, y, x)
        self.parameters["m"] = self.parameters["m"] - lr * grad["dm"]
        self.parameters["c"] = self.parameters["c"] - lr * grad["dc"]

    def _epoch(self, x, y, lr, verbose: bool, n: int):
        pred = self.predict(x)
        cost = self._cost(pred, y)
        self._back_prop(pred, y, x, lr)
        if verbose:
            print("Epoch: {}, loss: {}".format(n, cost))

    def train(self, x, y, lr=0.0000005, epochs=100, verbose: bool = True):
        self.parameters["m"] = np.random.uniform(0, 1) * -1
        self.parameters["c"] = np.random.uniform(0, 1) * -1

        for i in range(epochs):
            self._epoch(x, y, lr, verbose, n=i)


x = data["GRE Score"]
y = data["Chance of Admit "]

print(x.shape, y.shape)

reg = Regression("linear")
reg.train(x, y)
