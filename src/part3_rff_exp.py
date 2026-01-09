import os
import torch
import numpy as np

from dataset import get_mnist_binary_regression
from models import RFFModel
from trainer import train_and_evaluate


# ============================================================
# Experiment configuration
# ============================================================

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dataset size (đủ lớn để double descent rõ)
N_TRAIN = 1000
N_TEST = 5000

INPUT_DIM = 784
SIGMA = 5.0  # bandwidth Gaussian kernel

OUTPUT_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "part3_rff.txt")


# ============================================================
# Load data
# ============================================================

X_train, y_train, X_test, y_test = get_mnist_binary_regression(
    n_train=N_TRAIN,
    n_test=N_TEST,
    digits=(0, 1),
    seed=SEED
)

print("MNIST binary regression loaded.")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ============================================================
# Sweep number of random features
# ============================================================

# Sweep rất dày quanh interpolation threshold (~ N_TRAIN)
num_features_list = (
    list(range(50, 300, 25)) +
    list(range(300, 800, 10)) +
    list(range(800, 1500, 50)) +
    list(range(1500, 3000, 250))
)

num_features_list = sorted(set(num_features_list))


results = []

for D in num_features_list:
    print(f"Training RFF with D = {D}")

    model = RFFModel(
        input_dim=INPUT_DIM,
        num_features=D,
        sigma=SIGMA,
        seed=SEED
    )

    train_mse, test_mse = train_and_evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    results.append((D, train_mse, test_mse))

    print(
        f"D={D:4d} | train={train_mse:.4e} | test={test_mse:.4e}"
    )


# ============================================================
# Save results
# ============================================================

with open(OUTPUT_FILE, "w") as f:
    f.write("num_features,train_mse,test_mse\n")
    for D, tr, te in results:
        f.write(f"{D},{tr},{te}\n")

print(f"Results saved to {OUTPUT_FILE}")