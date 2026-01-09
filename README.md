# Thực nghiệm Double Descent với Linear Model và Random Fourier Features

Repository này chứa mã nguồn thực nghiệm hiện tượng **double descent** trong các mô hình học máy quá tham số,
dựa trên các công trình của **Belkin et al.** và **Hastie et al.**

Toàn bộ thực nghiệm sử dụng **Gradient Descent theo từng epoch (iterative training)**,
không sử dụng *lazy training*, nhằm phản ánh đúng **implicit bias của GD**
(dẫn đến nghiệm có norm nhỏ nhất như phân tích lý thuyết trong paper).

---

## Tính tái lập

- Kết quả **100% có thể tái lập**
- Cố định random seed
- Dataset được tải tự động
- Không có thành phần ngẫu nhiên không kiểm soát

---

## Cấu trúc repository

```
.
├── models.py              # Linear model và Random Fourier Features
├── dataset.py             # Tải và tiền xử lý dataset (MNIST / Fashion-MNIST)
├── part2_linear_exp.py    # Thực nghiệm double descent cho Linear Model
├── part3_rff_exp.py       # Thực nghiệm double descent cho RFF
├── plot_results.py        # Vẽ biểu đồ train / test error
├── run_experiments.sh     # Script chạy toàn bộ thực nghiệm
├── requirements.txt
├── logs/
└── plots/
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Chạy thực nghiệm

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

Sau khi chạy xong, các hình vẽ double descent (train error và test error)
sẽ được lưu trong thư mục `plots/`.
