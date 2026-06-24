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
from sklearn.neural_network import MLPClassifier

from lime.lime_tabular import LimeTabularExplainer
import weightedSHAP
# import shap
import TaylorPODA_engine
from tool_funcs import bootstrap_mean_ci

randomseed = 2026
secure_small = 0.00000000001
n_sample = 100

# -------------------------------- Data loading ------------------------------------------
# ['abalone', 'california', 'concrete']
datasets = ['abalone', 'california', 'concrete']
for dataset_name in datasets:
    dataset_df = pd.read_csv(f"datasets/{dataset_name}.csv")
    X = dataset_df.iloc[:, :-1]
    y = dataset_df.iloc[:,  -1]
    y, _ = stats.boxcox(y)

    # -------------------------------- Feature engineering ----------------------------------
    if dataset_name == 'abalone':
        categorical_features_list = X.select_dtypes(include=['object']).columns.values.tolist()
        X = pd.get_dummies(X, columns=categorical_features_list, prefix='Sex', drop_first=True)

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
    # task_model = MLPRegressor()
    with open(f'models/{dataset_name}_xgbr.pickle', 'rb') as ll:
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

    results_columns = ['inc_aup_occ1', 'inc_aup_lime', 'inc_aup_shap', 'inc_aup_wshap', 'inc_aup_tpoda',
                       'exc_aup_occ1', 'exc_aup_lime', 'exc_aup_shap', 'exc_aup_wshap', 'exc_aup_tpoda',
                       'inc_mse_occ1', 'inc_mse_lime', 'inc_mse_shap', 'inc_mse_wshap', 'inc_mse_tpoda',
                       'exc_mse_occ1', 'exc_mse_lime', 'exc_mse_shap', 'exc_mse_wshap', 'exc_mse_tpoda']
    results_df = pd.DataFrame(0.0, index=range(search_range), columns=results_columns)

    for index_selected_TBX_sample in range(search_range):  # search_range
        # range(search_range)
        # print(' ')
        print('================================== Processing {}/{} of {} dataset =============================='.format(
            index_selected_TBX_sample+1,search_range,dataset_name))

        X_TBX = Xtest[index_selected_TBX_sample:index_selected_TBX_sample+1]
        X_TBX_nd = X_TBX.to_numpy()
        ypred_TBX = task_model_callable.predict(X_TBX)
        y_avg = optimiser.masked_calculator.compute_masked_output(X_TBX, np.zeros(X_TBX.shape[1]))
        if hasattr(y_avg, "item"):
            y_avg = y_avg.item()

        # n_nearest_poda = min(round(0.3 * X_background.shape[0]), 50)
        # distances = np.linalg.norm(X_background - X_TBX_nd, axis=1)
        # nearest_indices = np.argsort(distances)[:n_nearest_poda]
        # X_support_PODA = X_background[nearest_indices]

        # print('')
        # print('-------- Analyzing TaylorPODA_engine explanation --------')
        optimised_attribution_inc_aup = optimiser.generate_optimised_attribution(
            input=X_TBX, options=16, dirichlet_scale=1, withMaskModel=1, approx=False, obj='inc_aup')
        a_tpoda_inc_aup = np.round(optimised_attribution_inc_aup['optimized attribution'], 8)
        # inc_aup_tpoda = round(optimised_attribution_inc_aup['objective score'], 3)
        _, inc_aup_tpoda = optimiser.present_inc_aup(a_tpoda_inc_aup, X_TBX)

        optimised_attribution_exc_aup = optimiser.generate_optimised_attribution(
            input=X_TBX, options=16, dirichlet_scale=1, withMaskModel=1, approx=False, obj='exc_aup')
        a_tpoda_exc_aup = np.round(optimised_attribution_exc_aup['optimized attribution'], 8)
        _, exc_aup_tpoda = optimiser.present_exc_aup(a_tpoda_exc_aup, X_TBX)

        optimised_attribution_inc_mse = optimiser.generate_optimised_attribution(
            input=X_TBX, options=16, dirichlet_scale=1, withMaskModel=1, approx=False, obj='inc_mse')
        a_tpoda_inc_mse = np.round(optimised_attribution_inc_mse['optimized attribution'], 8)
        _, inc_mse_tpoda = optimiser.present_inc_mse(a_tpoda_inc_mse, X_TBX)

        optimised_attribution_exc_mse = optimiser.generate_optimised_attribution(
            input=X_TBX, options=16, dirichlet_scale=1, withMaskModel=1, approx=False, obj='exc_mse')
        a_tpoda_exc_mse = np.round(optimised_attribution_exc_mse['optimized attribution'], 8)
        _, exc_mse_tpoda = optimiser.present_exc_mse(a_tpoda_exc_mse, X_TBX)

        results_df.loc[index_selected_TBX_sample, 'inc_aup_tpoda'] = inc_aup_tpoda
        results_df.loc[index_selected_TBX_sample, 'exc_aup_tpoda'] = exc_aup_tpoda
        results_df.loc[index_selected_TBX_sample, 'inc_mse_tpoda'] = inc_mse_tpoda
        results_df.loc[index_selected_TBX_sample, 'exc_mse_tpoda'] = exc_mse_tpoda

        # print('')
        # print('-------- Analyzing WeightedSHAP explanation --------')
        exp_dict = weightedSHAP.compute_attributions(
            problem, ML_model, task_model_callable, conditional_extension, Xtrain_wshap, ytrain_wshap, Xest_wshap,
            yest_wshap, X_TBX_nd, ytest[index_selected_TBX_sample:index_selected_TBX_sample+1], obj='inc_aup')
        a_wshap_inc_aup = np.array(exp_dict['value_list']).reshape(-1, 1)
        _, inc_aup_wshap = optimiser.present_inc_aup(a_wshap_inc_aup, X_TBX)

        exp_dict = weightedSHAP.compute_attributions(
            problem, ML_model, task_model_callable, conditional_extension, Xtrain_wshap, ytrain_wshap, Xest_wshap,
            yest_wshap, X_TBX_nd, ytest[index_selected_TBX_sample:index_selected_TBX_sample+1], obj='exc_aup')
        a_wshap_exc_aup = np.array(exp_dict['value_list']).reshape(-1, 1)
        _, exc_aup_wshap = optimiser.present_exc_aup(a_wshap_exc_aup, X_TBX)

        exp_dict = weightedSHAP.compute_attributions(
            problem, ML_model, task_model_callable, conditional_extension, Xtrain_wshap, ytrain_wshap, Xest_wshap,
            yest_wshap, X_TBX_nd, ytest[index_selected_TBX_sample:index_selected_TBX_sample+1], obj='inc_mse')
        a_wshap_inc_mse = np.array(exp_dict['value_list']).reshape(-1, 1)
        _, inc_mse_wshap = optimiser.present_inc_mse(a_wshap_inc_mse, X_TBX)

        exp_dict = weightedSHAP.compute_attributions(
            problem, ML_model, task_model_callable, conditional_extension, Xtrain_wshap, ytrain_wshap, Xest_wshap,
            yest_wshap, X_TBX_nd, ytest[index_selected_TBX_sample:index_selected_TBX_sample+1], obj='exc_mse')
        a_wshap_exc_mse = np.array(exp_dict['value_list']).reshape(-1, 1)
        _, exc_mse_wshap = optimiser.present_exc_mse(a_wshap_exc_mse, X_TBX)

        results_df.loc[index_selected_TBX_sample, 'inc_aup_wshap'] = inc_aup_wshap
        results_df.loc[index_selected_TBX_sample, 'exc_aup_wshap'] = exc_aup_wshap
        results_df.loc[index_selected_TBX_sample, 'inc_mse_wshap'] = inc_mse_wshap
        results_df.loc[index_selected_TBX_sample, 'exc_mse_wshap'] = exc_mse_wshap

        # print('')
        # print('---------- Analyzing SHAP explanation ----------')
        # Here we directly extract the Shapley results calculated during TaylorPODA_engine, as it is within one of its possible solutions.
        # In this way, we avoid re-calculating Shapley explanation, and secure a comparable setting for other explanation methods.
        # Ref: https://github.com/ykwon0407/WeightedSHAP/blob/main/notebook/Example_fraud_inclusion_AUC.ipynb

        optimised_attribution_Shapley = optimiser.generate_optimised_attribution(
            input=X_TBX, options='Shapley', withMaskModel=1)
        a_shap= np.round(optimised_attribution_Shapley['Shapley attribution'], 8)
        _, inc_aup_shap = optimiser.present_inc_aup(a_shap, X_TBX)
        _, exc_aup_shap = optimiser.present_exc_aup(a_shap, X_TBX)
        _, inc_mse_shap = optimiser.present_inc_mse(a_shap, X_TBX)
        _, exc_mse_shap = optimiser.present_exc_mse(a_shap, X_TBX)
        results_df.loc[index_selected_TBX_sample, 'inc_aup_shap'] = inc_aup_shap
        results_df.loc[index_selected_TBX_sample, 'exc_aup_shap'] = exc_aup_shap
        results_df.loc[index_selected_TBX_sample, 'inc_mse_shap'] = inc_mse_shap
        results_df.loc[index_selected_TBX_sample, 'exc_mse_shap'] = exc_mse_shap

        # print('')
        # print('-------- Analyzing Occlusion-1 explanation --------')
        occ1_attribution = optimiser.generate_occ1_attribution(input=X_TBX)
        a_occ1 = np.round(occ1_attribution, 8)
        _, inc_aup_occ1 = optimiser.present_inc_aup(a_occ1, X_TBX)
        _, exc_aup_occ1 = optimiser.present_exc_aup(a_occ1, X_TBX)
        _, inc_mse_occ1 = optimiser.present_inc_mse(a_occ1, X_TBX)
        _, exc_mse_occ1 = optimiser.present_exc_mse(a_occ1, X_TBX)
        results_df.loc[index_selected_TBX_sample, 'inc_aup_occ1'] = inc_aup_occ1
        results_df.loc[index_selected_TBX_sample, 'exc_aup_occ1'] = exc_aup_occ1
        results_df.loc[index_selected_TBX_sample, 'inc_mse_occ1'] = inc_mse_occ1
        results_df.loc[index_selected_TBX_sample, 'exc_mse_occ1'] = exc_mse_occ1

        # print('')
        # print('---------- Analyzing LIME explanation ----------')
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
        _, inc_aup_lime = optimiser.present_inc_aup(a_lime, X_TBX)
        _, exc_aup_lime = optimiser.present_exc_aup(a_lime, X_TBX)
        _, inc_mse_lime = optimiser.present_inc_mse(a_lime, X_TBX)
        _, exc_mse_lime = optimiser.present_exc_mse(a_lime, X_TBX)
        results_df.loc[index_selected_TBX_sample, 'inc_aup_lime'] = inc_aup_lime
        results_df.loc[index_selected_TBX_sample, 'exc_aup_lime'] = exc_aup_lime
        results_df.loc[index_selected_TBX_sample, 'inc_mse_lime'] = inc_mse_lime
        results_df.loc[index_selected_TBX_sample, 'exc_mse_lime'] = exc_mse_lime

        print(f'Debugging')

    print(f'Tests for {dataset_name} is done')
    results_df.to_csv(f'results/v2026_xgbr_{dataset_name}_results_.csv', index=False)
    print('Results saved')

for dataset_name in datasets:
    results_df = pd.read_csv(f'results/v2026_xgbr_{dataset_name}_results_.csv')
    # results_df = results_df[[col for col in results_df.columns if ('_shap' in col) or ('_tpoda' in col)]]
    results_df_shap_ref = pd.DataFrame(index=results_df.index)

    reverse_metrics = ('inc_aup', 'exc_auc', 'inc_mse')

    for col in results_df.columns:
        prefix = "_".join(col.split("_")[:2])
        shap_col = f"{prefix}_shap"

        if shap_col in results_df.columns:

            if any(metric in col for metric in reverse_metrics):
                diff = results_df[shap_col] - results_df[col]
            else:
                diff = results_df[col] - results_df[shap_col]

            results_df_shap_ref[col] = (
                    diff
                    # / (results_df[shap_col] + 1e-4)
                    * 100
            ).round(2)

    summary_df = bootstrap_mean_ci(results_df_shap_ref, n_bootstrap=100, ci=95, random_state=randomseed)
    mse_mask = summary_df.index.str.contains('mse')
    summary_df.loc[mse_mask] *= 100
    summary_df = summary_df.round(1)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(f'=======================Result of {dataset_name}: =======================================')
    print(summary_df)
    print('=========================================================================================')

print('Completed')
