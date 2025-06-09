import numpy as np
from numba import njit
import pandas as pd
import torch
import random
import math

class SubsetGenerator:
    def __init__(self, d):
        """
        Initialize the subset generation module
        :param d: Number of features (dimension)
        """
        self.d = d

    def generate_random_subset(self):
        """
        Randomly generates a {0, 1}^d vector
        :return: A random vector of {0, 1}^d
        """
        return np.random.choice([0, 1], size=self.d)

class MaskedCalculator:
    def __init__(self, model, background_data):
        """
        Initialize the masked calculation module
        :param model: The model function f(x)
        :param background_data: Background data B
        """
        self.model = model
        self.background_data = background_data

    def compute_masked_output(self, input, subset_vector):
        """
        Compute the masked output based on the subset vector
        :param input: input sample (dataframe)
        :param subset_vector: A {0,1}^d vector that determines which features are masked
        :return: The masked model output f(x_S)
        """
        if isinstance(input, np.ndarray):
            input_np = input
        else:
            input_np = input.to_numpy()

        if isinstance(subset_vector, np.ndarray):
            subset_vector_np = subset_vector
        else:
            subset_vector_np = np.array(subset_vector)

        amount_background = self.background_data.shape[0]
        # Create the masked input
        masked_output = 0.0
        for i in range(amount_background):
            masked_input = np.where(subset_vector_np == 1, input_np, self.background_data[i,:])
            # print(i, '-th background sample; masked_input: ', masked_input, 'masked_output: ', masked_output)
            out = self.model(masked_input)
            if hasattr(out, "item") and out.ndim == 0:
                out = out.item()
            masked_output += out
            # print('exact masked input: ', masked_input, '; exact masked output: ', out)

        # print('masked input: ', masked_input, '; masked output: ', masked_output/amount_background)
        # Compute the model output
        return masked_output/amount_background

class MaskedCalculator_withModel:
    def __init__(self, model, maskModel):
        """
        Initialize the masked calculation module
        :param model: The model function f(x)
        :param maskModel: A surrogate model to generate masked output
        """
        self.model = model
        self.maskModel = maskModel

    def compute_masked_output(self, input, subset_vector):
        """
        Compute the masked output based on the subset vector
        :param input: input sample, ndarray ideally
        :param subset_vector: A {0,1}^d vector that determines which features are masked, ndarray ideally
        :return: The masked model output f(x_S)
        """
        if isinstance(input, np.ndarray):
            input_np = input
        else:
            input_np = input.to_numpy()

        if isinstance(subset_vector, np.ndarray):
            subset_vector_np = subset_vector
        else:
            subset_vector_np = np.array(subset_vector)
        subset_vector_np = subset_vector_np[np.newaxis, :]

        masked_output = self.maskModel(input_np, subset_vector_np)
        if hasattr(masked_output, "item"):
            masked_output = masked_output.item()

        return masked_output

class Taylor_PODA_explainer:
    def __init__(self, model, background_data, maskModel=None):
        """
        Initialize the Taylor-PODA explainer for the to-be-explained input sample
        :param input: The input of the to-be-explained sample (dataframe)
        :param model: The black-box task model
        :param background_data: The data samples to support generating an explanation (ndarray)
        """
        self.model = model
        self.background_data = background_data
        self.masked_calculator = MaskedCalculator(model, background_data)
        if maskModel is not None:
            self.masked_calculator = MaskedCalculator_withModel(model, maskModel)

    def calculate_attribution_v1(self, input, Harsanyi_weights=None, withMaskModel=0, dirichlet_scale=1):
        """
        Calculate the attribution value for each feature of the to-be-explained input
            v1: with improved calculate efficiency to avoid repeated masked output
        :param input: The to-be-explained input
        :param Harsanyi_weights: (dict) Can pass a series of designed weights, or uniformlly distributed by default
        :return: [a0, a1, ...]
        """

        if isinstance(input, np.ndarray):
            input_np = input
        else:
            input_np = input.to_numpy()
        if input_np.ndim == 1:
            dimension = input_np.shape[0]
        else:
            dimension = input_np.shape[1]

        set_allfeatures = (1,) * dimension

        sets_S = generate_combinations_numba(set_allfeatures)

        fx = self.model(input_np)
        if hasattr(fx, "item"):
            fx = fx.item()

        # Prepare all the to-be-used masked outputs and restore them into a dict
        MaskedOutputs_dict = {}
        for S in sets_S:
            if (np.sum(1-S)) == 0:
                f_masked = fx
                # print('S: ', S, ', f_masked = fx = ', f_masked)
            else:
                f_masked = self.masked_calculator.compute_masked_output(input_np, S)

            MaskedOutputs_dict[tuple(S)] = f_masked

        # print("MaskedOutputs_dict", MaskedOutputs_dict)

        # Prepare all the to-be-used harsanyi dividends using the restored masked outputs,
        # and restore them into a dict;
        Harsanyi_dict = {}
        for S in sets_S:
            Harsanyi_output = 0
            all_possible_T = generate_combinations_numba(S)
            for T in all_possible_T:
                flag = is_difference_odd_or_even(S, T)
                term = MaskedOutputs_dict[tuple(T)]
                Harsanyi_output += flag * term
            # Compute the model output
            Harsanyi_dict[tuple(S)] = Harsanyi_output

        if Harsanyi_weights is not None:
            W = Harsanyi_weights
        else:
            # Initialize weights matrix
            W = np.zeros((2 ** dimension, dimension))
            for S in sets_S:
                S_dec_index = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(S)))
                sum_S = sum(S)
                for i in range(dimension):
                    if S[i] == 1:
                        W[S_dec_index,i] = 1 - 1.0/sum_S  # essentially, Shapley
            # print("Unified(Shapley) weights: ", W)

        # Calculate the attribution values a0, a1, ...
        # print('PODA weights:', W)
        A = []

        for i in range(dimension):
            # print(f'Calculating {i}-th feature attribution')
            Si = [1, ] * dimension
            Si[i] = 0
            fxi = self.masked_calculator.compute_masked_output(input_np, Si)  # Model output masking out the i-th feature
            H_average = 0
            for S, H in Harsanyi_dict.items():
                if sum(S) > 1 and S[i] == 1:
                    S_dec_index = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(S)))
                    w = W[(S_dec_index,i)]
                    H_average += w * H
            a = fx - fxi - H_average
            # print('H_average_PODA: ', H_average)
            A.append(a)

        return A

    def calculate_attribution_approx(self, input, withMaskModel=0, rank=2, n_sample=10, dirichlet_scale=1):
        """
        Calculate the attribution value for each feature of the to-be-explained input
            v1: with improved calculate efficiency to avoid repeated masked output
        :param input: The to-be-explained input
        :param Harsanyi_weights: (dict) Can pass a series of designed weights, or uniformlly distributed by default
        :return: [a0, a1, ...]
        """

        if isinstance(input, np.ndarray):
            input_np = input
        else:
            input_np = input.to_numpy()
        if input_np.ndim == 1:
            dimension = input_np.shape[0]
        else:
            dimension = input_np.shape[1]

        set_allfeatures = (1,) * dimension

        fx = self.model(input_np)
        if hasattr(fx, "item"):
            fx = fx.item()

        # Calculate the attribution values a0, a1, ...
        # print('PODA weights:', W)

        MaskedOutputs_dict = {}
        Harsanyi_dict = {}
        for i in range(dimension):

            sets_S = generate_combinations_numba_truncation(set_allfeatures, i, rank=rank, n_sample=n_sample)

            for S in sets_S:
                if (np.sum(1 - S)) == 0:
                    f_masked = fx
                    # print('S: ', S, ', f_masked = fx = ', f_masked)
                else:
                    all_possible_sub_S = generate_combinations_numba(S)
                    for SS in all_possible_sub_S:
                        if tuple(SS) not in MaskedOutputs_dict:
                            f_masked = self.masked_calculator.compute_masked_output(input_np, SS)
                            MaskedOutputs_dict[tuple(SS)] = f_masked

            # print("MaskedOutputs_dict", MaskedOutputs_dict)

            # Prepare all the to-be-used harsanyi dividends using the restored masked outputs,
            # and restore them into a dict;
            for S in sets_S:
                if tuple(S) not in Harsanyi_dict:
                    Harsanyi_output = 0
                    all_possible_T = generate_combinations_numba(S)
                    for T in all_possible_T:
                        flag = is_difference_odd_or_even(S, T)
                        if tuple(T) not in MaskedOutputs_dict:
                            MaskedOutputs_dict[tuple(T)] = self.masked_calculator.compute_masked_output(input_np, T)
                        term = MaskedOutputs_dict[tuple(T)]
                        Harsanyi_output += flag * term
                    # Compute the model output
                    Harsanyi_dict[tuple(S)] = Harsanyi_output

        W_dict = {}
        for S, H in Harsanyi_dict.items():
            possible_weights = 1 - generate_dirichlet_weights_approx(dirichlet_scale, sum(S))
            W_dict[S] = possible_weights

        A = []
        for i in range(dimension):
            # print(f'Calculating {i}-th feature attribution')
            Si = [1, ] * dimension
            Si[i] = 0
            fxi = self.masked_calculator.compute_masked_output(input_np, Si)  # Model output masking out the i-th feature

            j = 2
            sum_wH = 0

            for j in range(rank+1):
                if j > 1:
                    n_i_in_H = 0
                    h = 0
                    for S, H in Harsanyi_dict.items():
                        if S[i] == 1 and sum(S) == j:
                            n_i_in_H += 1
                            k = sum(1 for kk in range(i + 1) if S[kk] == 1) - 1
                            h +=  W_dict[S][0, k] * H
                    h = h / n_i_in_H
                    sum_wH += h * (math.comb(dimension-1, j-1))

            a = fx - fxi - sum_wH
            # print('H_average_PODA: ', H_average)
            A.append(a)

        return A

    def calculate_attribution_LiteOne(self, input, LiteOne_weights=None, Shapley_attribution=None, H_ShapleyBasis=None):
        if isinstance(input, np.ndarray):
            input_np = input
        else:
            input_np = input.to_numpy()
        if input_np.ndim == 1:
            dimension = input_np.shape[0]
        else:
            dimension = input_np.shape[1]

        set_allfeatures = (1,) * dimension
        sets_S = generate_combinations_numba(set_allfeatures)

        fx = self.model(input_np)
        if hasattr(fx, "item"):
            fx = fx.item()

        # Prepare all the to-be-used masked outputs and restore them into a dict
        MaskedOutputs_dict = {}
        for S in sets_S:
            if (np.sum(1-S)) == 0:
                f_masked = fx
                # print
            else:
                f_masked = self.masked_calculator.compute_masked_output(input_np, S)

            MaskedOutputs_dict[tuple(S)] = f_masked

        # Prepare all the to-be-used harsanyi dividends using the restored masked outputs,
        # and restore them into a dict;
        Harsanyi_dict = {}
        for S in sets_S:
            Harsanyi_output = 0
            all_possible_T = generate_combinations_numba(S)
            for T in all_possible_T:
                flag = is_difference_odd_or_even(S, T)
                term = MaskedOutputs_dict[tuple(T)]
                Harsanyi_output += flag * term
            # Compute the model output
            Harsanyi_dict[tuple(S)] = Harsanyi_output

        if Shapley_attribution is None:
            HS_array = np.zeros((dimension, 2))
            W = np.zeros((2 ** dimension, dimension))
            for S in sets_S:
                S_dec_index = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(S)))
                sum_S = sum(S)
                for i in range(dimension):
                    if S[i] == 1:
                        W[S_dec_index, i] = 1.0 / sum_S  # essentially, Shapley

            A = []

            for i in range(dimension):
                Si = [1, ] * dimension
                Si[i] = 0
                fxi = MaskedOutputs_dict[tuple(Si)]
                H_sum = 0
                H_shapley_weighted = 0
                for S, H in Harsanyi_dict.items():
                    if sum(S) > 1 and S[i] == 1:
                        S_dec_index = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(S)))
                        w = W[S_dec_index, i]
                        H_sum += H
                        H_shapley_weighted += w * H
                HS_array[i, 0] = H_sum
                HS_array[i, 1] = H_shapley_weighted
                a = fx - fxi - H_sum + H_shapley_weighted
                # print('Shapley HS_array[i, 0]: ', HS_array[i, 0])
                # print('Shapley HS_array[i, 1]: ', HS_array[i, 1])
                A.append(a)

        else:
            W = LiteOne_weights
            A = []
            HS_array = H_ShapleyBasis
            for i in range(dimension):
                Si = [1, ] * dimension
                Si[i] = 0
                fxi = MaskedOutputs_dict[tuple(Si)]
                # print('HS_array[i, 0]: ', HS_array[i, 0])
                # print('HS_array[i, 1]: ', HS_array[i, 1])
                a = fx - fxi - HS_array[i, 0] + HS_array[i, 1] * W[0, i]

                A.append(a)

        return A, HS_array

class Taylor_PODA_optimiser:
    def __init__(self, model, background_data, maskModel=None):
        """
        :param options: Optional solutions of the optimisation work, one of the options is the best answer
        :param dirichlet_scale: Default setting of dirichlet_scale to generate the weights by d-distribution
        """
        self.model = model
        self.background_data = background_data
        self.explainer = Taylor_PODA_explainer(model, background_data, maskModel=maskModel)
        if maskModel is not None:
            self.masked_calculator = MaskedCalculator_withModel(model, maskModel)
            # print('Mask model is triggered.')
        else:
            self.masked_calculator = MaskedCalculator(model, background_data)

    def generate_optimised_attribution(self, input, options=16, dirichlet_scale=1, withMaskModel=0, approx=False, rank=2, n_sample=10):
        """
        :param input: The to-be-explained input (dataframe)
        :return: An optimised attribution
        """

        if isinstance(input, np.ndarray):
            input_np = input
        else:
            input_np = input.to_numpy()
        if input_np.ndim == 1:
            dimension = input_np.shape[0]
        else:
            dimension = input_np.shape[1]

        set_allfeatures = (1,) * dimension

        if dimension >= 15:
            print('Calculation load can be overly heavy, please consider to configure -rank')
            approx = True

        if options == 'Shapley':
            # Set unified weights when i == -1,  which essentially lead to Shapley attribution.
            # Ref: Deng, Huiqi, et al. "Unifying fourteen post-hoc attribution methods with taylor interactions." IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).
            # So that to ensure the optimised TaylorPODA_engine result is better or equal than Shapley attribution in terms of better AUP.
            W = 'Shapley'
            attibution = self.explainer.calculate_attribution_v1(input_np, withMaskModel=withMaskModel)
            Shapley_attribution = attibution
            attribution_results_dict = dict()
            attribution_results_dict['Shapley attribution'] = Shapley_attribution

            return attribution_results_dict

        if (approx is False):
            sets_S = generate_combinations_numba(set_allfeatures)

            for i in range(options):
                # print("------ Traversing the randomly generated solutions [", i + 1, " / ", options,"] ------")
                W = np.zeros((2 ** dimension, dimension))
                for S in sets_S:
                    sum_S = sum(tuple(S))
                    if sum_S>1:
                        S_dec_index = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(S)))
                        possible_weights = self.generate_dirichlet_weights(dirichlet_scale, sum_S)
                        weight_indicator = 0
                        for j in range(dimension):
                            if S[j] != 0:
                                W[S_dec_index, j] = 1 - possible_weights[0, weight_indicator]
                                weight_indicator += 1
                attibution = self.explainer.calculate_attribution_v1(input_np, W, withMaskModel=withMaskModel)

                AUP_score = self.calculate_AUP(attibution, input_np)

                if i == 0:
                    best_attribution = attibution
                    best_weights = W
                    AUP_score_best = AUP_score
                else:
                    if AUP_score < AUP_score_best:
                        best_attribution = attibution
                        best_weights = W
                        AUP_score_best = AUP_score

            # print('PODA_weights:', W)
        else:
            for i in range(options):
                print("------ Traversing the randomly generated solutions [", i + 1, " / ", options,"] ------")

                attibution = self.explainer.calculate_attribution_approx(input_np, withMaskModel=withMaskModel, rank=rank, n_sample=n_sample, dirichlet_scale=dirichlet_scale)

                AUP_score = self.calculate_AUP(attibution, input_np)

                if i == 0:
                    best_attribution = attibution
                    best_weights = i
                    AUP_score_best = AUP_score
                else:
                    if AUP_score<AUP_score_best:
                        best_attribution = attibution
                        best_weights = i
                        AUP_score_best = AUP_score

            # print('Current_attribution = ', attibution)
            # print('Current_score = ', round(AUP_score, 6), '; Best score so far (smaller >>> better) = ', round(AUP_score_best, 6))

        # print('best_weights for PODA = ', best_weights)

        attribution_results_dict = dict()
        attribution_results_dict['Best weights'] = best_weights
        attribution_results_dict['Optimised attribution'] = best_attribution

        return attribution_results_dict

    def generate_occ1_attribution(self, input):
        if isinstance(input, np.ndarray):
            input_np = input
        else:
            input_np = input.to_numpy()
        if input_np.ndim == 1:
            dimension = input_np.shape[0]
        else:
            dimension = input_np.shape[1]
        A = []

        fx = self.model(input_np)
        if hasattr(fx, "item") and fx.ndim == 0:
            fx = fx.item()

        for i in range(dimension):
            S_fxi = np.ones(dimension)
            S_fxi[i] = 0
            fxi = self.masked_calculator.compute_masked_output(input_np, S_fxi)  # Model output masking out the i-th feature
            a = fx - fxi
            A.append(a)

        return A


    def calculate_AUP(self, attibution, input):

        if isinstance(input, np.ndarray):
            input_np = input
        else:
            input_np = input.to_numpy()
        if input_np.ndim == 1:
            dimension = input_np.shape[0]
        else:
            dimension = input_np.shape[1]

        indexed_attribution = list(enumerate(np.array(attibution)))
        sorted_attribution = sorted(indexed_attribution, key=lambda x: (abs(x[1]), x[0]), reverse=True)
        sorted_ranks = [x[0] for x in sorted_attribution]
        # print('Importance rank for each feature within this solution: ', sorted_ranks)

        attibution_len = len(attibution)
        rank = [0] * attibution_len
        for rank_index, (original_index, _) in enumerate(sorted_attribution):
            rank[original_index] = rank_index

        top_features_set = [0]*attibution_len
        score = 0

        fx = self.model(input_np)
        if hasattr(fx, "item") and fx.ndim == 0:
            fx = fx.item()

        for i in range(attibution_len-1):
            top_i_feature = rank.index(i)
            top_features_set[top_i_feature] = 1
            masked_output_without_nontop = self.masked_calculator.compute_masked_output(input_np, top_features_set)
            score += abs(fx - masked_output_without_nontop)

            # print('top_features_set: ', top_features_set)
            # print('masked_output_without_nontop: ', masked_output_without_nontop)

        return score

    def present_AUP(self, attibution, input):

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

        top_features_set = [0]*attibution_len
        errorlist_curent_top = []
        AUP = 0.0

        fx = self.model(input_np)
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

    def present_inc_auc(self, attibution, input):

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

        top_features_set = [0]*attibution_len
        auclist_curent_top = []
        auc = 0.0

        fx = self.model(input_np)
        if hasattr(fx, "item") and fx.ndim == 0:
            fx = fx.item()

        # print('AUC with top K important features: }')
        for i in range(attibution_len):
            top_i_feature = rank.index(i)
            top_features_set[top_i_feature] = 1
            # print('top_features_set: ', top_features_set)
            masked_output_without_nontop = self.masked_calculator.compute_masked_output(input_np, top_features_set)

            roc = int((fx > 0.5) == (masked_output_without_nontop > 0.5))

            if hasattr(roc, "item"):
                roc = roc.item()

            auc += roc
            # print('roc = ', roc)
            auclist_curent_top.append(auc)
            # print('auclist_curent_top = ', auclist_curent_top)

        auc = auc / attibution_len
        # print('Total auc = ', auc)

        return auclist_curent_top, auc


    def present_exc_auc(self, attibution, input):

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

        top_features_set = [1]*attibution_len
        auclist_curent_top = []
        auc = 0.0

        fx = self.model(input_np)
        if hasattr(fx, "item") and fx.ndim == 0:
            fx = fx.item()

        # print('AUC with top K important features: }')
        for i in range(attibution_len):
            top_i_feature = rank.index(i)
            top_features_set[top_i_feature] = 0
            # print('top_features_set: ', top_features_set)
            masked_output_without_nontop = self.masked_calculator.compute_masked_output(input_np, top_features_set)

            roc = int((fx > 0.5) == (masked_output_without_nontop > 0.5))

            if hasattr(roc, "item"):
                roc = roc.item()

            auc += roc
            # print('roc = ', roc)
            auclist_curent_top.append(auc)
            # print('auclist_curent_top = ', auclist_curent_top)

        auc = auc / attibution_len
        # print('Total auc = ', auc)

        return auclist_curent_top, auc

    def present_inc_mse(self, attibution, input):

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

        top_features_set = [0]*attibution_len
        errorlist_curent_top = []
        mse = 0.0

        fx = self.model(input_np)
        if hasattr(fx, "item") and fx.ndim == 0:
            fx = fx.item()

        # print('MSE with top K important features: }')
        for i in range(attibution_len):
            top_i_feature = rank.index(i)
            top_features_set[top_i_feature] = 1
            # print('top_features_set: ', top_features_set)
            masked_output_without_nontop = self.masked_calculator.compute_masked_output(input_np, top_features_set)
            error = (fx - masked_output_without_nontop) * (fx - masked_output_without_nontop)

            if hasattr(error, "item"):
                error = error.item()

            mse += error
            # print(MSE)
            errorlist_curent_top.append(mse)
            # print(MSE[0])

        mse = mse / attibution_len
        # print('Total MSE = ', mse)

        return errorlist_curent_top, mse

    def present_exc_mse(self, attibution, input):

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

        top_features_set = [1] * attibution_len
        errorlist_curent_top = []
        mse = 0.0

        fx = self.model(input_np)
        if hasattr(fx, "item") and fx.ndim == 0:
            fx = fx.item()

        # print('MSE with top K important features: }')
        for i in range(attibution_len):
            top_i_feature = rank.index(i)
            top_features_set[top_i_feature] = 0
            # print('top_features_set: ', top_features_set)
            masked_output_without_nontop = self.masked_calculator.compute_masked_output(input_np, top_features_set)
            error = (fx - masked_output_without_nontop) * (fx - masked_output_without_nontop)

            if hasattr(error, "item"):
                error = error.item()

            mse += error
            # print(MSE)
            errorlist_curent_top.append(mse)
            # print(MSE[0])

        mse = mse / attibution_len
        # print('Total MSE = ', mse)

        return errorlist_curent_top, mse

    def generate_dirichlet_weights(self, dirichlet_scale, amount_weights):
        weights_baseline = [dirichlet_scale] * amount_weights
        weights = np.random.dirichlet(weights_baseline, size=1)
        return weights


def generate_dirichlet_weights_approx(dirichlet_scale, amount_weights):
    weights_baseline = [dirichlet_scale] * amount_weights
    weights = np.random.dirichlet(weights_baseline, size=1)
    return weights

def generate_combinations_numba(mask):
    """
    Given a mask (indicating the feature's presence or absence), generate all of its possible subsets (mask code).

    :param mask: The parent mask (tuple or array-like)
    :return: A 2D NumPy array of all possible subsets (mask code) of the parent mask
    """
    mask = np.array(mask)  # Ensure that the input is a NumPy array
    one_positions = np.where(mask == 1)[0]  # Find the positions of 1s
    n = len(one_positions)  # Count how many 1s are in the mask

    total_combinations = 2 ** n  # Total number of subsets (2^n)

    # Create a list to store the results
    results = []

    # Generate all possible subsets by iterating over the range of 0 to 2^n (total combinations)
    for i in range(total_combinations):
        combination = mask.copy()  # Start with a copy of the original mask
        # Determine which positions to keep as 1 (based on the binary representation of i)
        for j in range(n):
            if (i >> j) & 1:  # If the j-th bit of i is set, keep the corresponding position as 1
                combination[one_positions[j]] = 1
            else:
                combination[one_positions[j]] = 0
        results.append(combination)  # Store the combination in results

    return np.array(results)


@njit
def comb_count(n, k):
    """Compute combinations count C(n, k) using integer math."""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

def generate_combinations_numba_truncation(mask, feature, rank, n_sample):
    """
    Efficiently sample submasks from a binary mask with the following conditions:
    - Each submask must contain the specified `feature`
    - Each submask must contain at least 2 and at most `rank` active positions (1s)
    - All active positions must be a subset of the original active (1-valued) mask
    - For each subset size k in [2, rank], up to `n_sample` random non-repeating samples are taken

    Parameters:
        mask (array-like): Binary mask (e.g., [1, 0, 1, 1, 0])
        feature (int): Index of the feature that must be included in every sampled mask
        rank (int): Maximum number of active (1) positions per submask
        n_sample (int): Number of samples to generate per k

    Returns:
        np.ndarray: 2D array of sampled submasks, each including `feature` and meeting constraints
    """
    mask = np.array(mask)
    one_positions = np.where(mask == 1)[0]

    if feature not in one_positions:
        raise ValueError(f"The feature index {feature} is not active (1) in the input mask.")

    # Positions we can sample from, excluding the fixed feature
    available_positions = [pos for pos in one_positions if pos != feature]

    results = []

    for k in range(2, rank + 1):  # Only consider subsets of size >= 2
        if k > len(one_positions):
            break

        max_possible = comb_count(len(available_positions), k - 1)
        if max_possible == 0:
            continue

        seen = set()
        attempts = 0
        max_attempts = n_sample * 10  # Avoid infinite loops

        while len(seen) < min(n_sample, max_possible) and attempts < max_attempts:
            sampled = tuple(sorted(random.sample(available_positions, k - 1)))
            full_comb = tuple(sorted((feature,) + sampled))

            if full_comb not in seen:
                seen.add(full_comb)
                new_mask = np.zeros_like(mask)
                new_mask[list(full_comb)] = 1
                results.append(new_mask)

            attempts += 1

    return np.array(results)


def is_difference_odd_or_even(tuple1, tuple2):
    count1 = sum(tuple1)
    count2 = sum(tuple2)

    difference = abs(count1 - count2)

    return -1 if difference % 2 != 0 else 1  # -1: odd, 1: even
