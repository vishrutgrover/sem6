import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
X, y = datasets.make_moons(n_samples=500, noise=0.25, random_state=69)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Plot function
def plot(model, X, y):
    h = 0.02
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

# -------- Gaussian (RBF) Kernel --------
rbf_model = SVC(kernel='rbf')
rbf_model.fit(X_train, y_train)
rbf_pred = rbf_model.predict(X_test)
print("RBF Accuracy:", accuracy_score(y_test, rbf_pred))
plot(rbf_model, X_test, y_test)

# -------- Sigmoid Kernel --------
sig_model = SVC(kernel='sigmoid')
sig_model.fit(X_train, y_train)
sig_pred = sig_model.predict(X_test)
print("Sigmoid Accuracy:", accuracy_score(y_test, sig_pred))
plot(sig_model, X_test, y_test)