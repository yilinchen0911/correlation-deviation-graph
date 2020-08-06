from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


class EmpiricalSpatialCorrelationModel(object):
    """docstring for ."""

    def __init__(self, method, data=[None, None], search_range = (1, 300)):
        self.method = method
        self.dist = np.array(data[0]).reshape(-1, 1)
        self.coff = np.array(data[1])
        if self.method == "jayaram_baker_2009":
            def jayaram_baker_2009(T, h, isVs30clustered=0):
                if T < 1:
                    if isVs30clustered == 0:
                        b = 8.5 + 17.2 * T
                    elif isVs30clustered == 1:
                        b = 40.7 - 15.0 * T
                elif T >= 1:
                    b = 22.0 + 3.7 * T
                rho = np.exp(-3 * h / b)
                return rho
            self.predict = jayaram_baker_2009

        if self.method == "fitted_exponential":
            def fitted_exponential(T, h, isVs30clustered=0):
                "y = exp(-x/b)"
                positveCoffIndex = self.coff > 0
                reg = LinearRegression(fit_intercept=False).fit(
                    self.dist[positveCoffIndex], np.log(self.coff[positveCoffIndex]))
                self.range = float((-1 / reg.coef_) * 3)
                return np.exp(reg.predict(np.array(h).reshape(-1, 1)))
            self.predict = fitted_exponential

        if self.method == "weighted_least_square":
            decay_rate = 5
            search_precision = 0.1
            weights = np.exp(-self.dist / decay_rate)
            search_candidate = np.arange(search_range[0], search_range[1], search_precision)
            loss = []
            for range_candidate in search_candidate:
                coff_hat = self.exponential_range(self.dist, range_candidate)
                coff_hat = coff_hat.reshape(-1)
                loss_range_candidate = np.sum(weights * (self.coff - coff_hat) ** 2)
                loss.append(loss_range_candidate)
            # plt.plot(search_candidate, loss)
            # plt.show()
            self.range = search_candidate[np.argmin(loss)]
            def weighted_least_square(T, h, isVs30clustered=0):
                return self.exponential_range(h, self.range)
            self.predict = weighted_least_square

        if self.method == "constant_exponential":
            def constant_exponential(T, h, isVs30clustered=0):
                return np.exp(np.array(h)/-450)
            self.predict = constant_exponential

    def exponential_range(self, h, range):
        return np.exp(-3 * h / range)

# empiricalModel = empiricalSpatialCorrelationModel("fitted_exponential", data = [[5, 10, 20, 30, 40, 80], [0.75, 0.6, 0.5, 0.25, 0.2, -0.1]])
# plt.plot(np.arange(0, 100, 0.01), empiricalModel.predict(1, np.arange(0, 100, 0.01)), '-r', label = 'Empirical Prediction')
# plt.show()
