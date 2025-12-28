import torch
import numpy as np
from torchvision import datasets, transforms


# ============================================================
# Part 2: Synthetic Linear Regression Dataset
# ============================================================

def make_random_feature_regression(
    n_train=400,
    n_test=10000,
    input_dim=20,
    noise_std=0.1,
    seed=0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    w_star = torch.randn(input_dim)

    X_train = torch.randn(n_train, input_dim)
    y_train = X_train @ w_star + noise_std * torch.randn(n_train)

    X_test = torch.randn(n_test, input_dim)
    y_test = X_test @ w_star + noise_std * torch.randn(n_test)

    return X_train, y_train, X_test, y_test


# ============================================================
# Part 3: Binary MNIST (0 vs 1) for RFF Regression
# ============================================================

def get_mnist_binary_regression(
    n_train=1000,
    n_test=5000,
    digits=(0, 1),
    seed=0
):
    """
    MNIST binary regression task (0 vs 1)
    Labels:
        digit_0 -> -1
        digit_1 -> +1

    Dùng cho RFF double descent (Part 3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    def filter_binary(dataset, n_samples):
        X_list, y_list = [], []
        count = 0
        for x, y in dataset:
            if y in digits:
                label = -1.0 if y == digits[0] else 1.0
                X_list.append(x)
                y_list.append(torch.tensor(label))
                count += 1
                if count >= n_samples:
                    break
        return torch.stack(X_list), torch.stack(y_list)

    X_train, y_train = filter_binary(train_dataset, n_train)
    X_test, y_test = filter_binary(test_dataset, n_test)

    return X_train, y_train, X_test, y_test
