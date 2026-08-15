from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(X, Y, test_size=0.2, random_state=42):
    # Stratification preserves the target-class balance in both splits.
    return train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)

def scale_data(X_train, X_test):
    # Fit only on training data to prevent test-set information leaking into training.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
