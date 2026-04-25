import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_data():
    X, y = datasets.make_moons(
        n_samples=500,
        noise=0.25,
        random_state=69
    )
    return train_test_split(X, y, test_size=0.3, random_state=69)


def plot_boundary(model, X, y, title, color1='#4C72B0', color2='#DD8452'):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', linewidths=0.4, s=40)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=ax, ticks=[0, 1], label='Class')
    plt.tight_layout()
    plt.show()


def gaussian_svm(X_train, X_test, y_train, y_test):
    print("=" * 40)
    print("  Experiment 1: Gaussian (RBF) Kernel")
    print("=" * 40)

    model = SVC(kernel="rbf")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Correct  : {int(acc * len(y_test))} / {len(y_test)}")
    print("=" * 40 + "\n")

    plot_boundary(model, X_test, y_test, "Gaussian (RBF) Kernel — Decision Boundary")
    return model


def sigmoid_svm(X_train, X_test, y_train, y_test):
    print("=" * 40)
    print("  Experiment 2: Sigmoid Kernel")
    print("=" * 40)

    model = SVC(kernel="sigmoid")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Correct  : {int(acc * len(y_test))} / {len(y_test)}")
    print("=" * 40 + "\n")

    plot_boundary(model, X_test, y_test, "Sigmoid Kernel — Decision Boundary")
    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    gaussian_svm(X_train, X_test, y_train, y_test)
    sigmoid_svm(X_train, X_test, y_train, y_test)
