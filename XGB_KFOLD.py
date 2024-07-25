import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
 

train = pd.read_parquet('train.parquet')
val = pd.read_parquet('val.parquet')



scaler = StandardScaler()

n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900]
max_depth = [3, 5, 7]
learning_rate =  [0.1, 0.01, 0.001, 0.0001]

rmspes = {}


data = pd.concat([train, val])

from sklearn.model_selection import KFold

# Assuming your data is stored in the 'data' variable
X = data.drop(columns='target')
y = data['target']

# Create a KFold object with 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the folds and split the data
for i,(train_index, test_index) in enumerate(kfold.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
        # Perform your model training and evaluation on each fold
        # ...
    dtrain_reg = xgb.DMatrix(X_train, y_train)
    dtest_reg = xgb.DMatrix(X_val, y_val)
        # Rest of your code for training and evaluation
    for n in tqdm(n_estimators):
        for depth in max_depth:
            for l in learning_rate:
                params = {"objective": "reg:squarederror", "tree_method": "gpu_hist", "max_depth": depth, "learning_rate": l}
                model = xgb.train(
                    params=params,
                    dtrain=dtrain_reg,
                    num_boost_round=n,
                    )
                preds = model.predict(dtest_reg)
                rmspe = np.sqrt(np.mean(((y_val - preds) / y_val) ** 2))
                if i == 0:
                    rmspes[f"{n}_{depth}_{l}"] = rmspe
                else:
                    rmspes[f"{n}_{depth}_{l}"] = (rmspes[f"{n}_{depth}_{l}"]*i +rmspe)/(i+1)

    
with open('Output2/rmspes10sec_withprice_withtest_Kfold', 'wb') as file:
    pickle.dump(rmspes, file)
print("finished")
