import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import auc
import xgboost as xgb

randomseed = 2024
secure_small = 0.00000000001


for dataset_name in ['breast_cancer_cls', 'rice_cls', 'titanic_cls']:
    # ['banknote_cls', 'breast_cancer_cls', 'diabetes_cls', 'rice_cls', 'tic_tac_toe_endgame_cls', 'titanic_cls']

    # -------------------------------- Data loading ------------------------------------------
    # print('Please select datasets: \n[banknote_cls, breast_cancer_cls, diabetes_cls, rice_cls, tic_tac_toe_endgame_cls, titanic_cls]')
    # dataset_name = input('Type and enter: ')
    print("=========================================================================================")
    print(f"Processing dataset: {dataset_name}")
    dataset_df = pd.read_csv(f"datasets/{dataset_name}.csv")
    X = dataset_df.iloc[:, :-1]
    y = dataset_df['label']

    # -------------------------------- Feature engineering ----------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values.tolist())
    y_normalized = y
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=randomseed)

    np.random.seed(randomseed)
    indices = np.random.choice(len(Xtest), size=100, replace=False)
    Xtest = Xtest.iloc[indices]
    ytest = ytest.iloc[indices]
    np.random.seed(None)

    ytest = ytest.values.ravel()
    train = pd.concat([Xtrain, ytrain], axis=1)

    # -------------------------------- Model building ---------------------------------------
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=randomseed
    )
    model.fit(Xtrain, ytrain.values.ravel())
    task_model = model

    ypred0 = task_model.predict_proba(Xtest)[:, 1]

    with open (f'models/{dataset_name}_xgbc.pickle', 'wb') as ww:
        pickle.dump(task_model, ww)

    task_model = xgb.XGBClassifier()
    with open (f'models/{dataset_name}_xgbc.pickle', 'rb') as ll:
        task_model = pickle.load(ll)

    ypred = task_model.predict_proba(Xtest)[:, 1]


print('Debug')
