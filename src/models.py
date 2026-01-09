import torch
import torch.nn as nn
import math


# ============================================================
# Part 2: Linear Regression Model (Iterative / Epoch-based)
# ============================================================

class LinearRegressionModel(nn.Module):
    """
    Linear regression on random ReLU features.
    Trained via SGD/Adam (Iterative).
    """
    def __init__(self, input_dim, num_features, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        
        # 1. Fixed Random Features (Layer 1)
        # Weight cố định, không update (requires_grad=False)
        self.W = nn.Parameter(torch.randn(num_features, input_dim), requires_grad=False)
        
        # 2. Learnable Readout (Layer 2)
        # Bias=False để đúng tính chất random feature regression thuần túy
        self.readout = nn.Linear(num_features, 1, bias=False)
        
        # Lưu num_features để dùng cho normalization
        self.num_features = num_features

    def forward(self, x):
        # x: (Batch, Input_dim)
        
        # Feature map: phi(x) = ReLU(Wx^T) / sqrt(D)
        # Chia cho sqrt(D) là cực kỳ quan trọng để ổn định gradient khi D lớn
        phi = torch.relu(x @ self.W.T) / math.sqrt(self.num_features)
        
        # Output: (Batch,)
        return self.readout(phi).squeeze()



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
