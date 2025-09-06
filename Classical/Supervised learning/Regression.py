import numpy as np
import pandas as pd

import Cost
import Regularization


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
        return np.dot(x, m) + c

    def _cost(self, pred, y, cost_f):
        return cost_f.calculate(y, pred)

    def _grad(self, pred, y, x):
        derivatives = {}
        dc = 2 * np.mean((pred - y))
        dm = 2 * np.mean((pred - y) * x, axis=0)
        derivatives["dm"] = dm
        derivatives["dc"] = dc
        return derivatives

    def _back_prop(self, pred, y, x, lr):
        grad = self._grad(pred, y, x)
        self.parameters["m"] = self.parameters["m"] - lr * grad["dm"]
        self.parameters["c"] = self.parameters["c"] - lr * grad["dc"]

    def _epoch(
        self,
        x,
        y,
        lr,
        verbose: bool,
        n: int,
        cost_f: Cost,
        regularization: Regularization,
    ):
        pred = self.predict(x)
        cost = self._cost(pred, y, cost_f)
        self._back_prop(pred, y, x, lr)
        if verbose:
            print("Epoch: {}, loss: {}".format(n, cost))

    def train(
        self,
        x,
        y,
        cost: Cost,
        regularization: Regularization = None,
        lr=0.000005,
        epochs=1500,
        verbose: bool = True,
    ):
        if not isinstance((x, y), (np.ndarray, np.ndarray)):
            x = x.to_numpy()
            y = y.to_numpy()
        self.parameters["m"] = np.random.rand(x.shape[1], 1) * -1
        self.parameters["c"] = np.random.uniform(0, 1) * -1

        for i in range(epochs):
            self._epoch(
                x, y, lr, verbose, n=i, cost_f=cost, regularization=regularization
            )


x = data.drop(["Chance of Admit "], axis=1)

y = data[["Chance of Admit "]]

reg = Regression(type="linear")
reg.train(x, y, cost=Cost.MSE())
