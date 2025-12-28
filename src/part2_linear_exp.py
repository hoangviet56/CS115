import os
from dataset import make_random_feature_regression
from models import RandomFeatureLinearModel

N_TRAIN = 400
N_TEST = 10000
INPUT_DIM = 20
NOISE_STD = 0.1
SEED = 0

X_train, y_train, X_test, y_test = make_random_feature_regression(
    n_train=N_TRAIN,
    n_test=N_TEST,
    input_dim=INPUT_DIM,
    noise_std=NOISE_STD,
    seed=SEED
)

num_features_list = (
    list(range(20, 200, 10)) +
    list(range(200, 600, 5)) +
    list(range(600, 1200, 50))
)

results = []

for D in num_features_list:
    model = RandomFeatureLinearModel(
        input_dim=INPUT_DIM,
        num_features=D,
        seed=SEED
    )

    model.fit(X_train, y_train)

    train_mse = model.mse(X_train, y_train)
    test_mse = model.mse(X_test, y_test)

    results.append((D, train_mse, test_mse))

    print(
        f"D={D:4d} | train={train_mse:.3e} | test={test_mse:.3e}"
    )

with open("logs/part2_linear.txt", "w") as f:
    f.write("dim,train_mse,test_mse\n")
    for d, tr, te in results:
        f.write(f"{d},{tr},{te}\n")
