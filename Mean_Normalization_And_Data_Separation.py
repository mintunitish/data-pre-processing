import numpy as np

X = np.random.randint(0, 5001, (1000, 20))

ave_cols = X.mean(axis=0)
std_cols = X.std(axis=0)

X_norm = (X - ave_cols) / std_cols

row, col = X_norm.shape
row_indices = np.random.permutation(row)

X_train = X_norm[row_indices[0:600]]
X_crossVal = X_norm[row_indices[600:800]]
X_test = X_norm[row_indices[800:]]

print(X_train.shape)
print(X_crossVal.shape)
print(X_test.shape)

