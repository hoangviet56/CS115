import os
import csv
import matplotlib.pyplot as plt


# ============================================================
# Utility: load result file
# ============================================================

def load_results(file_path):
    dims, train_mse, test_mse = [], [], []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dims.append(int(row[list(row.keys())[0]]))
            train_mse.append(float(row["train_mse"]))
            test_mse.append(float(row["test_mse"]))

    return dims, train_mse, test_mse


# ============================================================
# Plot double descent (2 subplots)
# ============================================================

def plot_double_descent(
    result_file,
    title,
    output_path,
    interpolation_threshold=None
):
    dims, train_mse, test_mse = load_results(result_file)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---------------- Train ----------------
    axes[0].plot(dims, train_mse, marker="o")
    axes[0].set_title("Train Error")
    axes[0].set_xlabel("Model Dimension")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True)

    # ---------------- Test ----------------
    axes[1].plot(dims, test_mse, marker="o")
    axes[1].set_title("Test Error")
    axes[1].set_xlabel("Model Dimension")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True)

    # Interpolation threshold (n_train)
    if interpolation_threshold is not None:
        for ax in axes:
            ax.axvline(
                interpolation_threshold,
                linestyle="--",
                linewidth=1
            )

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Plot saved to {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Part 2: Linear Regression
    plot_double_descent(
        result_file="logs/part2_linear.txt",
        title="Part 2: Linear Regression Double Descent",
        output_path="plots/part2_double_descent.png",
        interpolation_threshold=200
    )

    # Part 3: Random Fourier Features
    plot_double_descent(
        result_file="logs/part3_rff.txt",
        title="Part 3: Random Fourier Features Double Descent",
        output_path="plots/part3_rff_double_descent.png",
        interpolation_threshold=1000
    )
