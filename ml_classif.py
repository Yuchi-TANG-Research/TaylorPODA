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
from sklearn.neural_network import MLPClassifier

randomseed = 2024
secure_small = 0.00000000001


for dataset_name in ['titanic_cls']:
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

    act = 'logistic' if dataset_name in ['banknote_cls', 'breast_cancer_cls', 'diabetes_cls'] else 'tanh'

    # -------------------------------- Model building ---------------------------------------
    param_grid = {
        'hidden_layer_sizes': [(64, 64, 32, 16), (32, 32, 32, 16), (64, 32, 32), (32, 32, 16)],
        'activation': [act]
    }
    mlp = MLPClassifier(random_state=randomseed)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=2)
    grid_search.fit(Xtrain, ytrain.values.ravel())
    print("Best Parameters:", grid_search.best_params_)
    print("Best ROC-AUC Score on Training Set:", grid_search.best_score_)
    task_model = grid_search.best_estimator_

    ypred0 = task_model.predict_proba(Xtest)[:, 1]

    with open (f'models/{dataset_name}_MLPC.pickle', 'wb') as ww:
        pickle.dump(task_model, ww)

    # Best Parameters:

    # {'activation': 'logistic', 'hidden_layer_sizes': (64, 32, 32)} banknote_cls
    # {'activation': 'logistic', 'hidden_layer_sizes': (32, 32, 16)} breast_cancer_cls
    # {'activation': 'logistic', 'hidden_layer_sizes': (64, 32, 32)} diabetes_cls
    # {'activation': 'tanh', 'hidden_layer_sizes': (64, 32, 32)} rice_cls
    # {'activation': 'tanh', 'hidden_layer_sizes': (32, 32, 32, 16)} tic_tac_toe_endgame_cls
    # {'activation': 'tanh', 'hidden_layer_sizes': (64, 64, 32, 16)} titanic_cls

    task_model = MLPClassifier()
    with open (f'models/{dataset_name}_MLPC.pickle', 'rb') as ll:
        task_model = pickle.load(ll)

    ypred = task_model.predict_proba(Xtest)[:, 1]


print('Debug')
