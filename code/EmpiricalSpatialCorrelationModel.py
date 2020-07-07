from sklearn.linear_model import LinearRegression
import numpy as np


class EmpiricalSpatialCorrelationModel(object):
    """docstring for ."""

    def __init__(self, method, data=[None, None]):
        self.method = method
        self.dist = np.array(data[0]).reshape(-1, 1)
        self.coff = np.array(data[1])
        if self.method == "jayaram_baker_2009":
            def correlationModel(T, h, isVs30clustered=0):
                if T < 1:
                    if isVs30clustered == 0:
                        b = 8.5 + 17.2 * T
                    elif isVs30clustered == 1:
                        b = 40.7 - 15.0 * T
                elif T >= 1:
                    b = 22.0 + 3.7 * T
                rho = np.exp(-3 * h / b)
                return rho
        if self.method == "fitted_exponential":
            def correlationModel(T, h, isVs30clustered=0):
                "y = exp(-x/b)"
                positveCoffIndex = self.coff > 0
                reg = LinearRegression(fit_intercept=False).fit(
                    self.dist[positveCoffIndex], np.log(self.coff[positveCoffIndex]))
                return np.exp(reg.predict(np.array(h).reshape(-1, 1)))
        if self.method == "constant_exponential":
            def correlationModel(T, h, isVs30clustered=0):
                return np.exp(np.array(h)/-450)
        self.predict = correlationModel
# empiricalModel = empiricalSpatialCorrelationModel("fitted_exponential", data = [[5, 10, 20, 30, 40, 80], [0.75, 0.6, 0.5, 0.25, 0.2, -0.1]])
# plt.plot(np.arange(0, 100, 0.01), empiricalModel.predict(1, np.arange(0, 100, 0.01)), '-r', label = 'Empirical Prediction')
# plt.show()
