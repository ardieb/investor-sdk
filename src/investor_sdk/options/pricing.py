import abc
import numpy as np

from scipy.optimize import minimize, approx_fprime


class _BinomialTreeOptionPricingModel(abc.ABC):

    def __init__(self, delta: int):
        self.delta = delta
    
    def _calculate_option_value(self, factor: int, K: float, T: float, S: float, O: float, R: float) -> float:
        dt = self.delta
        N = int(T / dt)

        u = np.exp(O * np.sqrt(dt))

        p = (u - np.exp(-R * dt)) / (np.power(u, 2) - 1)
        q = np.exp(-R * dt) - p

        P = [0] * (N + 1)
        for i in range(N + 1):
            P[i] = max(0, factor * (S * np.power(u, 2*i - N) - K))
        
        for j in range(N - 1, -1, -1):
            for i in range(0, j+1):
                P[i] = p * P[i+1] + q * P[i]
                E = factor * (S * np.power(u, 2 * i - j) - K)
                P[i] = max(P[i], E, 0)
        
        return P[0]

    @abc.abstractmethod
    def calculate_option_value(self, K: float, T: float, S: float, R: float) -> float:
        raise NotImplementedError
    
    def implied_volatility(self, P: float, K: float, T: float, S: float, R: float) -> float:
        def target(V):
            return np.square(self.calculate_option_value(K, T, S, V, R) - P)

        guess = 0.5
        fprime = lambda x: approx_fprime(x, target, 1e-6)

        return minimize(target, guess, method='newton-cg', jac=fprime)


class BinomialTreeCallOptionPricingModel(_BinomialTreeOptionPricingModel):

    def calculate_option_value(self, K: float, T: float, S: float, O: float, R: float) -> float:
        return self._calculate_option_value(1, K, T, S, O, R)


class BinomialTreePutOptionPricingModel(_BinomialTreeOptionPricingModel):

    def calculate_option_value(self, K: float, T: float, S: float, O: float, R: float) -> float:
        return self._calculate_option_value(-1, K, T, S, O, R)
