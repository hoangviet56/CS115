import torch
import math


# ============================================================
# Linear Regression Model (Closed-form, Part 2)
# ============================================================

class LinearRegressionModel:
    """
    Linear regression với nghiệm closed-form:
        w = argmin ||Xw - y||^2

    Dùng để khảo sát double descent theo số chiều feature.
    """

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        """
        X: (n, d)
        y: (n,)
        """
        # Moore–Penrose pseudo-inverse để xử lý cả over/under-parameterized
        X_pinv = torch.linalg.pinv(X)
        self.w = X_pinv @ y

    def predict(self, X):
        return X @ self.w

    def mse(self, X, y):
        y_pred = self.predict(X)
        return torch.mean((y_pred - y) ** 2).item()
    
class RandomFeatureLinearModel:
    """
    Linear regression on random Gaussian features:
        phi(x) = ReLU(Wx) or Gaussian random features
    """

    def __init__(self, input_dim, num_features, seed=0):
        torch.manual_seed(seed)
        self.W = torch.randn(num_features, input_dim)
        self.w = None

    def features(self, X):
        return torch.relu(X @ self.W.T)

    def fit(self, X, y):
        Phi = self.features(X)
        self.w = torch.linalg.pinv(Phi) @ y

    def predict(self, X):
        return self.features(X) @ self.w

    def mse(self, X, y):
        return torch.mean((self.predict(X) - y) ** 2).item()



# ============================================================
# Random Fourier Features Model (Part 3)
# ============================================================

class RFFModel:
    """
    Random Fourier Features cho Gaussian kernel:

        phi(x) = sqrt(2/D) * cos(Wx + b)

    Sau đó giải regression tuyến tính trên feature space.
    """

    def __init__(self, input_dim, num_features, sigma=1.0, seed=0):
        self.input_dim = input_dim
        self.num_features = num_features
        self.sigma = sigma

        torch.manual_seed(seed)

        # Random frequencies
        self.W = torch.randn(num_features, input_dim) / sigma

        # Random phase
        self.b = 2 * math.pi * torch.rand(num_features)

        self.w = None  # weight của linear regression

    def _rff_features(self, X):
        """
        X: (n, input_dim)
        return: (n, num_features)
        """
        Z = X @ self.W.T + self.b
        return math.sqrt(2.0 / self.num_features) * torch.cos(Z)

    def fit(self, X, y):
        """
        Closed-form least squares trên RFF feature
        """
        Phi = self._rff_features(X)
        Phi_pinv = torch.linalg.pinv(Phi)
        self.w = Phi_pinv @ y

    def predict(self, X):
        Phi = self._rff_features(X)
        return Phi @ self.w

    def mse(self, X, y):
        y_pred = self.predict(X)
        return torch.mean((y_pred - y) ** 2).item()
