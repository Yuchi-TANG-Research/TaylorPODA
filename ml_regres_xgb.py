import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

randomseed = 2024
secure_small = 0.00000000001


for dataset_name in ['abalone', 'california', 'concrete']:

    # -------------------------------- Data loading ------------------------------------------
    # print('Please select datasets: \n[banknote_cls, breast_cancer_cls, diabetes_cls, rice_cls, tic_tac_toe_endgame_cls, titanic_cls]')
    # dataset_name = input('Type and enter: ')
    print("=========================================================================================")
    print(f"Processing dataset: {dataset_name}")
    dataset_df = pd.read_csv(f"datasets/{dataset_name}.csv")
    X = dataset_df.iloc[:, :-1]
    y = dataset_df.iloc[:, -1]
    y, _ = stats.boxcox(y)

    # -------------------------------- Feature engineering ----------------------------------
    if dataset_name in ['abalone', 'aging']:
        categorical_features_list = X.select_dtypes(include=['object']).columns.values.tolist()
        X = pd.get_dummies(X, columns=categorical_features_list, prefix='Sex', drop_first=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values.tolist())
    y_normalized = pd.DataFrame(scaler.fit_transform(y.reshape(-1, 1)))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=randomseed)

    np.random.seed(randomseed)
    indices = np.random.choice(len(Xtest), size=100, replace=False)
    Xtest = Xtest.iloc[indices]
    ytest = ytest.iloc[indices]
    np.random.seed(None)

    ytest = ytest.values.ravel()
    train = pd.concat([Xtrain, ytrain], axis=1)

    # -------------------------------- Model building ---------------------------------------
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=randomseed
    )
    model.fit(Xtrain, ytrain.values.ravel())
    task_model = model

    ypred0 = task_model.predict(Xtest)

    with open (f'models/{dataset_name}_xgbr.pickle', 'wb') as ww:
        pickle.dump(task_model, ww)

    task_model = xgb.XGBRegressor()
    with open (f'models/{dataset_name}_xgbr.pickle', 'rb') as ll:
        task_model = pickle.load(ll)

    ypred = task_model.predict(Xtest)


print('Debug')
