import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
from scipy import stats
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from lime.lime_tabular import LimeTabularExplainer
import weightedSHAP
import TaylorPODA_engine

randomseed = 2024
secure_small = 0.00000000001
n_sample = 100
index_selected_TBX_sample = 0

# -------------------------------- Data loading ------------------------------------------
dataset_name = 'concrete'
dataset_df = pd.read_csv(f"datasets/{dataset_name}.csv")
X = dataset_df.iloc[:, :-1]
y = dataset_df.iloc[:, -1]
y, _ = stats.boxcox(y)

# -------------------------------- Feature engineering ----------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.values.tolist())
y_normalized = pd.DataFrame(scaler.fit_transform(y.reshape(-1, 1)))
Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=randomseed)

np.random.seed(randomseed)
indices = np.random.choice(len(Xtest), size=n_sample, replace=False)
Xtest = Xtest.iloc[indices]
ytest = ytest.iloc[indices]
np.random.seed(None)

ytest = ytest.values.ravel()
train = pd.concat([Xtrain, ytrain], axis=1)

# -------------------------------- Model building ---------------------------------------
task_model = MLPRegressor()
with open (f'models/{dataset_name}_MLPR.pickle', 'rb') as ll:
    task_model = pickle.load(ll)

ypred = task_model.predict(Xtest)

# --------------------- Model varifying ------------------------
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

class ModelWrapper:
    def __init__(self, model):
        self.model = model
    def __call__(self, data):
        prediction = self.model.predict(data)
        return prediction
    def predict(self, data):
        return self.model.predict(data)

task_model_callable = ModelWrapper(task_model)

# --------------------- Results presenting ------------------------
Xbackground = Xtrain
X_background = Xbackground.to_numpy()

train_wshap, est_wshap = train_test_split(train, test_size=0.3, random_state=randomseed)
Xtrain_wshap = train_wshap.iloc[:, :-1].to_numpy()
ytrain_wshap = train_wshap.iloc[:, -1]
Xest_wshap = est_wshap.iloc[:, :-1].to_numpy()
yest_wshap = est_wshap.iloc[:, -1]
problem = 'regression'
ML_model = 'MLP'
search_range = Xtest.shape[0]  # Xtest.shape[0]

print('Configuring explaining environment:')

conditional_extension = weightedSHAP.generate_coalition_function(
    task_model_callable, Xtrain_wshap, Xest_wshap, problem, ML_model)

optimiser = TaylorPODA_engine.Taylor_PODA_optimiser(task_model_callable, X_background, maskModel=conditional_extension)

X_TBX = Xtest[index_selected_TBX_sample:index_selected_TBX_sample+1]
X_TBX_nd = X_TBX.to_numpy()
ypred_TBX = task_model.predict(X_TBX).item()
y_avg = optimiser.masked_calculator.compute_masked_output(X_TBX, np.zeros(X_TBX.shape[1]))
if hasattr(y_avg, "item"):
    y_avg = y_avg.item()

print('')
print('-------- Analyzing TaylorPODA_engine explanation --------')
optimised_attribution = optimiser.generate_optimised_attribution(
    input=X_TBX, options=16, dirichlet_scale=1, withMaskModel=1)
a_PODA = np.round(optimised_attribution['Optimised attribution'], 8)
_, aup_tpoda = optimiser.present_AUP(a_PODA, X_TBX)
_, incmse_tpoda = optimiser.present_inc_mse(a_PODA, X_TBX)
print(f'aup_tpoda = {aup_tpoda}, incmse_tpoda = {incmse_tpoda}')

print('')
print('-------- Analyzing WeightedSHAP explanation --------')
exp_dict = weightedSHAP.compute_attributions(
    problem, ML_model, task_model_callable, conditional_extension, Xtrain_wshap, ytrain_wshap, Xest_wshap,
    yest_wshap, X_TBX_nd, ytest[index_selected_TBX_sample:index_selected_TBX_sample+1])
a_wshap = np.array(exp_dict['value_list']).reshape(-1, 1)
_, aup_wshap = optimiser.present_AUP(a_wshap, X_TBX)
_, incmse_wshap = optimiser.present_inc_mse(a_wshap, X_TBX)
print(f'aup_wshap = {aup_wshap}, incmse_wshap = {incmse_wshap}')

print('')
print('---------- Analyzing SHAP explanation ----------')
optimised_attribution_Shapley = optimiser.generate_optimised_attribution(
    input=X_TBX, options='Shapley', withMaskModel=1)
a_shap= np.round(optimised_attribution_Shapley['Shapley attribution'], 8)
_, aup_shap = optimiser.present_AUP(a_shap, X_TBX)
_, incmse_shap = optimiser.present_inc_mse(a_shap, X_TBX)
print(f'aup_shap = {aup_shap}, incmse_shap = {incmse_shap}')

print('')
print('-------- Analyzing Occlusion-1 explanation --------')
occ1_attribution = optimiser.generate_occ1_attribution(input=X_TBX)
a_occ1 = np.round(occ1_attribution, 8)
_, aup_occ1 = optimiser.present_AUP(a_occ1, X_TBX)
_, incmse_occ1 = optimiser.present_inc_mse(a_occ1, X_TBX)
print(f'aup_occ1 = {aup_occ1}, incmse_occ1 = {incmse_occ1}')

print('')
print('---------- Analyzing LIME explanation ----------')
feature_list = X_TBX.columns.tolist()
explainer_lime = LimeTabularExplainer(
    training_data=Xbackground.values,
    training_labels=ytrain.loc[Xbackground.index].values,
    mode='regression',
    feature_names=feature_list,
    random_state=randomseed
)
a_lime_output = explainer_lime.explain_instance(X_TBX_nd[0], task_model_callable)
y_lime = a_lime_output.local_pred.item()
a_lime = [next((value for key, value in dict(a_lime_output.as_list()).items() if feature in key), None) for feature in feature_list]
lime_attribution_df = pd.DataFrame([a_lime], columns=feature_list)
a_lime = np.array(a_lime).reshape(-1, 1)
a_lime = np.where(a_lime == None, 0.0, a_lime)
a_lime = a_lime.astype(float)
_, aup_lime = optimiser.present_AUP(a_lime, X_TBX)
_, incmse_lime = optimiser.present_inc_mse(a_lime, X_TBX)
print(f'aup_lime = {aup_lime}, incmse_lime = {incmse_lime}')

