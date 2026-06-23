import numpy as np
import pandas as pd

def calculate_aup(model, attibution, input):
    if isinstance(input, np.ndarray):
        input_np = input
    else:
        input_np = input.to_numpy()
    if input_np.ndim == 1:
        dimension = input_np.shape[0]
    else:
        dimension = input_np.shape[1]

    indexed_attribution = list(enumerate(np.array(attibution)))
    # print('indexed_attribution: ', indexed_attribution)
    # print('indexed_attribution: ', type(indexed_attribution))
    sorted_attribution = sorted(indexed_attribution, key=lambda x: (abs(x[1]), x[0]), reverse=True)
    sorted_ranks = [x[0] for x in sorted_attribution]
    # print('Importance rank for each feature within this solution: ', sorted_ranks)

    attibution_len = len(attibution)
    rank = [0] * attibution_len
    for rank_index, (original_index, _) in enumerate(sorted_attribution):
        rank[original_index] = rank_index

    top_features_set = [0] * attibution_len
    errorlist_curent_top = []
    AUP = 0.0

    fx = model(input_np)
    if hasattr(fx, "item") and fx.ndim == 0:
        fx = fx.item()

    # print('AUP with top K important features: }')
    for i in range(attibution_len):
        top_i_feature = rank.index(i)
        top_features_set[top_i_feature] = 1
        # print('top_features_set: ', top_features_set)
        masked_output_without_nontop = self.masked_calculator.compute_masked_output(input_np, top_features_set)
        error = abs(fx - masked_output_without_nontop)

        if hasattr(error, "item"):
            error = error.item()

        AUP += error
        # print(AUP)
        errorlist_curent_top.append(AUP)
        # print(AUP[0])
    # print('Total AUP = ', AUP)

    return errorlist_curent_top, AUP

def bootstrap_mean_ci(results_df, n_bootstrap=100, ci=95, random_state=2026):
    rng = np.random.default_rng(random_state)

    values = results_df.to_numpy()
    n_rows = len(results_df)

    boot_means = []

    for _ in range(n_bootstrap):
        sampled_idx = rng.choice(n_rows, size=n_rows, replace=True)
        sampled_values = values[sampled_idx, :]
        boot_means.append(sampled_values.mean(axis=0))

    boot_means = np.array(boot_means)

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha, axis=0)
    upper = np.percentile(boot_means, 100 - alpha, axis=0)
    mean = values.mean(axis=0)

    summary_df = pd.DataFrame({
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
    }, index=results_df.columns)

    return summary_df
