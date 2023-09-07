import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Buat dataset
X, y = make_classification(n_samples=500, n_features=20, n_redundant=0, n_informative=15, random_state=42)

# Normalisasi
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_elm(X_train, y_train, hidden_neurons=500, lambda_=1e-5):
    input_weights = np.random.normal(size=[X_train.shape[1], hidden_neurons])
    biases = np.random.normal(size=[hidden_neurons])
    H = sigmoid(np.dot(X_train, input_weights) + biases)
    H_T = np.transpose(H)
    output_weights = np.dot(np.linalg.pinv(np.dot(H_T, H) + lambda_ * np.identity(hidden_neurons)), np.dot(H_T, y_train))
    return input_weights, biases, output_weights


def predict_elm(X, input_weights, biases, output_weights):
    H = sigmoid(np.dot(X, input_weights) + biases)
    return np.dot(H, output_weights)

# ELM (optimized)
hidden_neurons = 1500
lambda_ = 1e-4
input_weights, biases, output_weights = train_elm(X_train_val, y_train_val, hidden_neurons, lambda_)
y_pred_test = predict_elm(X_test, input_weights, biases, output_weights)
y_pred_test_bin = (y_pred_test > 0.5).astype(int)

print("ELM Results (optimized):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test_bin)}")
print(f"AUC: {roc_auc_score(y_test, y_pred_test)}")
print(f"F1 Score: {f1_score(y_test, y_pred_test_bin)}")
print("------------------------------------------------")

# Logistic Regression
lr = LogisticRegression(random_state=42).fit(X_train_val, y_train_val)
y_pred_test = lr.predict(X_test)
y_pred_test_prob = lr.predict_proba(X_test)[:, 1]

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
print(f"AUC: {roc_auc_score(y_test, y_pred_test_prob)}")
print(f"F1 Score: {f1_score(y_test, y_pred_test)}")
print("------------------------------------------------")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42).fit(X_train_val, y_train_val)
y_pred_test = dt.predict(X_test)

print("Decision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
print(f"AUC: {roc_auc_score(y_test, y_pred_test)}")
print(f"F1 Score: {f1_score(y_test, y_pred_test)}")
print("------------------------------------------------")

# Random Forest
rf = RandomForestClassifier(random_state=42).fit(X_train_val, y_train_val)
y_pred_test = rf.predict(X_test)
y_pred_test_prob = rf.predict_proba(X_test)[:, 1]

print("Random Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
print(f"AUC: {roc_auc_score(y_test, y_pred_test_prob)}")
print(f"F1 Score: {f1_score(y_test, y_pred_test)}")



# ELM Results (optimized):
# Accuracy: 0.92
# AUC: 0.9827378562826175
# F1 Score: 0.9199999999999999
# ------------------------------------------------
# Logistic Regression Results:
# Accuracy: 0.9
# AUC: 0.9642713769570453
# F1 Score: 0.9038461538461539
# ------------------------------------------------
# Decision Tree Results:
# Accuracy: 0.76
# AUC: 0.7663588920112405
# F1 Score: 0.7446808510638298
# ------------------------------------------------
# Random Forest Results:
# Accuracy: 0.89
# AUC: 0.9642713769570453
# F1 Score: 0.8952380952380953
