import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import get_fashion_mnist_binary
from models import LinearRegressionModel
import time

# ============================================================
# Configuration (FASHION-MNIST)
# ============================================================

N_TRAIN = 1000   
N_TEST = 5000
INPUT_DIM = 784
CLASSES = (4, 6) # Coat vs Shirt

# --- CẤU HÌNH MỚI ĐỂ LÀM MƯỢT ---
BASE_SEED = 42   # Seed cơ sở
N_REPEATS = 5    # Số lần chạy lặp lại tại mỗi điểm D để lấy trung bình
# --------------------------------

EPOCHS = 4000
LR = 1e-3 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running Linear Regression on FASHION-MNIST {CLASSES}")
print(f"Averaging over {N_REPEATS} runs per dimension.")
print(f"Interpolation expected at D ~ {N_TRAIN}")

# ============================================================
# Load Data
# ============================================================
# Load dữ liệu một lần, dùng seed cố định để đảm bảo train/test split giống nhau
X_train_raw, y_train_raw, X_test_raw, y_test_raw = get_fashion_mnist_binary(
    n_train=N_TRAIN, n_test=N_TEST, classes=CLASSES, seed=BASE_SEED
)
X_test = X_test_raw.to(DEVICE)
y_test = y_test_raw.to(DEVICE)

# ============================================================
# Sweep Range
# ============================================================
range_low = list(range(50, 800, 100))
range_peak = list(range(800, 1500, 50)) 
range_high = list(range(1500, 6000, 500))
num_features_list = sorted(list(set(range_low + range_peak + range_high)))

results = []
os.makedirs("logs", exist_ok=True)

# ============================================================
# Main Loop (Averaging Version)
# ============================================================

total_start_time = time.time()

for D in num_features_list:
    train_mse_accumulator = 0.0
    test_mse_accumulator = 0.0
    
    print(f"\nStarting D={D:4d}, running {N_REPEATS} times...")
    
    # Vòng lặp lặp lại để lấy trung bình
    for repeat in range(N_REPEATS):
        # Tạo seed khác nhau cho mỗi lần lặp
        current_seed = BASE_SEED + repeat
        
        # Cần đảm bảodataloader hoặc cách shuffle dữ liệu cũng dùng seed này
        # Ở đây ta dùng cách đơn giản là shuffle thủ công X_train
        torch.manual_seed(current_seed)
        perm = torch.randperm(X_train_raw.size(0))
        X_train = X_train_raw[perm].to(DEVICE)
        y_train = y_train_raw[perm].to(DEVICE)

        # 1. Init Model với current_seed
        model = LinearRegressionModel(
            input_dim=INPUT_DIM,
            num_features=D,
            seed=current_seed # Quan trọng: Khởi tạo weights khác nhau
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

        # 2. Training Loop
        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            preds = model(X_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 3. Evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            test_pred = model(X_test)
            train_mse_accumulator += criterion(train_pred, y_train).item()
            test_mse_accumulator += criterion(test_pred, y_test).item()
            
        # (Optional) In dấu chấm để biết tiến độ
        print(".", end="", flush=True)

    # Tính trung bình sau N_REPEATS
    avg_train_mse = train_mse_accumulator / N_REPEATS
    avg_test_mse = test_mse_accumulator / N_REPEATS

    results.append((D, avg_train_mse, avg_test_mse))

    print(f"\nFinished D={D:4d} | Avg Train MSE={avg_train_mse:.4f} | Avg Test MSE={avg_test_mse:.4f}")

total_end_time = time.time()
print(f"Total experiment time: {total_end_time - total_start_time:.2f}s")

# ============================================================
# Save Results
# ============================================================
with open("logs/part2_linear_avg.txt", "w") as f:
    f.write("dim,train_mse,test_mse\n")
    for d, tr, te in results:
        f.write(f"{d},{tr},{te}\n")
print("Done. Results saved to logs/part2_linear_avg.txt")