# Thuc nghiem Double Descent voi Linear Model va Random Fourier Features

Repository nay chua ma nguon thuc nghiem hien tuong double descent trong cac mo hinh hoc may qua tham so, dua tren cac cong trinh cua Belkin et al. va Hastie et al.

Toan bo thuc nghiem su dung Gradient Descent theo tung epoch (iterative training), khong su dung lazy training, nham phan anh dung implicit bias cua GD.

## Tinh tai lap
- Ket qua 100% co the tai lap
- Co dinh random seed
- Dataset tai tu dong
- Khong co thanh phan ngau nhien khong kiem soat

## Cau truc repository
.
├── models.py
├── dataset.py
├── part2_linear_exp.py
├── part3_rff_exp.py
├── plot_results.py
├── run_experiments.sh
├── requirements.txt
├── logs/
└── plots/

## Cai dat
pip install -r requirements.txt

## Chay thuc nghiem
chmod +x run_experiments.sh
./run_experiments.sh
