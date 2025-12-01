import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import seaborn as sns

# get data
def prep_data():
    # debug = True
    
    print("reading data...")
    d = pd.read_csv('hour.csv')
    
    # make feat
    print("making features...")
    
    d['h_sin'] = np.sin(2 * np.pi * d['hr'] / 24)  
    d['h_cos'] = np.cos(2 * np.pi * d['hr'] / 24)
    d['m_sin'] = np.sin(2 * np.pi * d['mnth'] / 12)   
    d['m_cos'] = np.cos(2 * np.pi * d['mnth'] / 12)
    
    # choose features i think good
    f = [
        'temp', 'atemp', 'hum', 'windspeed',  # basic
        'h_sin', 'h_cos', 'm_sin', 'm_cos',   # time
        'holiday', 'workingday', 'weathersit'  # other
    ]
    
    X = d[f].to_numpy() 
    
    # try make more features
    temp = X[:, 0]    # temp 
    hum = X[:, 2]    # hum
    wind = X[:, 3]    # wind
    
    
    X = np.column_stack([
        X,
        temp * hum,     
        temp * wind,    
        temp ** 2,    
        hum ** 2,     
        wind ** 2      
    ])
    
    y = d['cnt'].to_numpy() 
    
    # split train test
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, 
        y, 
        test_size=0.15, 
        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, 
        y_tr, 
        test_size=0.176, 
        random_state=42)
    
    # make nums small
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Performance Evaluation Metric：check model good or bad
def calc_score(y_true, y_pred): 
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse) 
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return mse, rmse, r2

# basic model：linear Regression
class LR:
    def __init__(self):
        self.w = None
        self.b = None
    
    def train(self, X, y):
        n = X.shape[0]
        X_b = np.c_[np.ones((n, 1)), X]
        
        print("solving...")
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self.b = self.theta[0]
        self.w = self.theta[1:]
        print("done!")
    
    def pred(self, X):
        return X @ self.w + self.b

# Gradient Descent model
class GD:
    def __init__(self, lr=0.05, epochs=300, lam=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.w = None
        self.b = None
        self.loss_hist = []  # for plot
    
    def train(self, X, y):
        # init
        n_feat = X.shape[1]
        self.w = np.zeros(n_feat)
        self.b = 0
        
        # train loop
        print("training...")
        for e in range(self.epochs): 
            y_pred = self.pred(X)
            
            # L2 loss
            loss = np.mean((y_pred - y) ** 2) + self.lam * np.sum(self.w ** 2)
            self.loss_hist.append(loss)
            
            if e % 20 == 0:  # print less
                print(f"epoch {e}, loss = {loss:.2f}")
            
            # update w and b
            dw = 2 * X.T @ (y_pred - y) / len(y) + 2 * self.lam * self.w
            db = 2 * np.mean(y_pred - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # # early stop
            # if e > 0 and abs(self.loss_hist[-1] - self.loss_hist[-2]) < 0.01:
            #     print(f"early stop at epoch {e}")
            #     break
    
    def pred(self, X):
        return X @ self.w + self.b

# L1 Regularization model
class LASSO:
    def __init__(self, lr=0.01, epochs=300, lam=0.1):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam 
        self.w = None
        self.b = None
        self.loss_hist = []
    
    def train(self, X, y):
        n_feat = X.shape[1]
        self.w = np.zeros(n_feat)
        self.b = 0
        
        print("training lasso...")
        for e in range(self.epochs):
            y_pred = self.pred(X)
            
            # L1 loss
            loss = np.mean((y_pred - y) ** 2) + self.lam * np.sum(np.abs(self.w))
            self.loss_hist.append(loss)
            
            if e % 20 == 0:
                print(f"epoch {e}, loss = {loss:.2f}")
            
            # update
            dw = 2 * X.T @ (y_pred - y) / len(y) + self.lam * np.sign(self.w)
            db = 2 * np.mean(y_pred - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def pred(self, X):
        return X @ self.w + self.b

# Grid search
def tune_params(model_class, p_grid, X_train, y_train, X_val, y_val):
    best_r2 = float('-inf')
    best_p = None
    
    print(f"\ntuning parameters...")
    for p in p_grid:
        # try this param
        print(f"trying params: {p}")
        model = model_class(**p)
        model.train(X_train, y_train)
        
        # check how good it is
        val_pred = model.pred(X_val)
        r2 = calc_score(y_val, val_pred)[2]
        print(f"got R2: {r2:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_p = p
    
    print(f"best params found: {best_p}, R2: {best_r2:.4f}")
    return best_p

if __name__ == "__main__":
    # get data
    X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
    print("data shape:", X_train.shape)
    
    print("\ndata info:")
    print(f"train: {X_train.shape[0]} samples")
    print(f"val: {X_val.shape[0]} samples")
    print(f"test: {X_test.shape[0]} samples")
    
    # model 1 has no params to tune
    print("\ntraining model 1...")
    m1 = LR()
    m1.train(X_train, y_train)
    
    # tune model 2 params
    gd_params = [
        {'lr': 0.01, 'epochs': 300, 'lam': 0.1},
        {'lr': 0.05, 'epochs': 300, 'lam': 0.01},
        {'lr': 0.1, 'epochs': 300, 'lam': 0.001}
    ]
    best_gd = tune_params(GD, gd_params, X_train, y_train, X_val, y_val)
    
    # tune model 3 params
    lasso_params = [
        {'lr': 0.005, 'epochs': 300, 'lam': 0.1},
        {'lr': 0.01, 'epochs': 300, 'lam': 0.05},
        {'lr': 0.02, 'epochs': 300, 'lam': 0.01}
    ]
    best_lasso = tune_params(LASSO, lasso_params, X_train, y_train, X_val, y_val)
    
    # train with best params
    print("\ntraining model 2 with best params...")
    m2 = GD(**best_gd)
    m2.train(X_train, y_train)
    
    print("\ntraining model 3 with best params...")
    m3 = LASSO(**best_lasso)
    m3.train(X_train, y_train)
    
    # rest of the code stays the same...
    print("\ntesting...")
    
    # model 1
    p1_train = m1.pred(X_train)
    p1_val = m1.pred(X_val)
    p1_test = m1.pred(X_test)
    
    # model 2
    p2_train = m2.pred(X_train)
    p2_val = m2.pred(X_val)
    p2_test = m2.pred(X_test)
    
    # model 3
    p3_train = m3.pred(X_train)
    p3_val = m3.pred(X_val)
    p3_test = m3.pred(X_test)
    
    # print scores
    print("\nModel 1 scores:")
    print(f"Train - MSE: {calc_score(y_train, p1_train)[0]:.2f}, R2: {calc_score(y_train, p1_train)[2]:.4f}")
    print(f"Val   - MSE: {calc_score(y_val, p1_val)[0]:.2f}, R2: {calc_score(y_val, p1_val)[2]:.4f}")
    print(f"Test  - MSE: {calc_score(y_test, p1_test)[0]:.2f}, R2: {calc_score(y_test, p1_test)[2]:.4f}")
    
    print("\nModel 2 scores:")
    print(f"Train - MSE: {calc_score(y_train, p2_train)[0]:.2f}, R2: {calc_score(y_train, p2_train)[2]:.4f}")
    print(f"Val   - MSE: {calc_score(y_val, p2_val)[0]:.2f}, R2: {calc_score(y_val, p2_val)[2]:.4f}")
    print(f"Test  - MSE: {calc_score(y_test, p2_test)[0]:.2f}, R2: {calc_score(y_test, p2_test)[2]:.4f}")
    
    print("\nModel 3 scores:")
    print(f"Train - MSE: {calc_score(y_train, p3_train)[0]:.2f}, R2: {calc_score(y_train, p3_train)[2]:.4f}")
    print(f"Val   - MSE: {calc_score(y_val, p3_val)[0]:.2f}, R2: {calc_score(y_val, p3_val)[2]:.4f}")
    print(f"Test  - MSE: {calc_score(y_test, p3_test)[0]:.2f}, R2: {calc_score(y_test, p3_test)[2]:.4f}")
    
    # plots stay the same
    plt.figure(figsize=(15, 5))
    
    # Training Loss in GD and Lasso
    plt.subplot(1, 3, 1)
    plt.plot(m2.loss_hist, label='GD')
    plt.plot(m3.loss_hist, label='Lasso')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, p1_test, alpha=0.5, label='Model 1')
    plt.scatter(y_test, p2_test, alpha=0.5, label='Model 2')
    plt.scatter(y_test, p3_test, alpha=0.5, label='Model 3')
    
    plt.plot([0, 800], [0, 800], 'r--')

    plt.xlim(0, 800)   
    plt.ylim(0, 800) 
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title('Predictions vs True')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.bar(range(len(m3.w)), m3.w)
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.title('LASSO Weights')
    
    plt.tight_layout()
    plt.savefig('result.png')
    # plt.show()  # not show now