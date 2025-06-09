import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import math

import TaylorPODA_engine
import weightedSHAP

randomseed = 2024
num_0 = 3
num_1 = 8
downsample_length = 28
index = 11
present_top_k = round(downsample_length * downsample_length)  # downsample_length * downsample_length

# -------------------------------- Data loading ------------------------------------------
Xtrain = np.load("datasets/mnist/Xtrain.npy")
ytrain = np.load("datasets/mnist/ytrain.npy")
Xtest = np.load("datasets/mnist/Xtest.npy")
ytest = np.load("datasets/mnist/ytest.npy")

def filter(x, y, num_0=0, num_1=1):
    mask = (y == num_0) | (y == num_1)
    x, y = x[mask], y[mask]
    y = (y == num_1).astype(int)
    return x, y

Xtrain, ytrain = filter(Xtrain, ytrain, num_0=num_0, num_1=num_1)
Xtest, ytest = filter(Xtest, ytest, num_0=num_0, num_1=num_1)

original_length = Xtest.shape[1]

Xtrain_tensor = tf.expand_dims(Xtrain, axis=-1)
Xtest_tensor = tf.expand_dims(Xtest, axis=-1)
Xtrain_downsampled = tf.image.resize(Xtrain_tensor, size=[downsample_length, downsample_length], method='area')
Xtest_downsampled = tf.image.resize(Xtest_tensor, size=[downsample_length, downsample_length], method='area')
Xtrain_downsampled_np = tf.squeeze(Xtrain_downsampled, axis=-1).numpy()
Xtest_downsampled_np = tf.squeeze(Xtest_downsampled, axis=-1).numpy()

original_X_TBX = Xtest[index]
downsampled_X_TBX = Xtest_downsampled_np[index]

# -------------------------------- Feature engineering ----------------------------------
Xtrain_downsampled_np = Xtrain_downsampled_np.astype('float32') / 255.0
Xtest_downsampled_np = Xtest_downsampled_np.astype('float32') / 255.0
Xtrain_downsampled_np_flatten = Xtrain_downsampled_np.reshape(-1, downsample_length * downsample_length)
Xtest_downsampled_np_flatten = Xtest_downsampled_np.reshape(-1, downsample_length * downsample_length)

# -------------------------------- Model using---------------------------------------
task_model = MLPClassifier()
with open (f'models/mnist{num_0}{num_1}_model_d{downsample_length}_v3.pickle', 'rb') as ll:
    task_model = pickle.load(ll)
predictions = task_model.predict(Xtest_downsampled_np_flatten)

# -------------------------------- Explanation providing --------------------------------
train = np.hstack((Xtrain_downsampled_np_flatten, ytrain.reshape(-1, 1)))
np.random.seed(randomseed)
background = train[np.random.choice(train.shape[0], 1000, replace=False)]
np.random.seed(None)
Xbackground = background[:, :-1]
ybackground = background[:, -1]
recover_factor = original_length // downsample_length
X_TBX = Xtest_downsampled_np_flatten[index:index+1]
y_TBX = task_model.predict(X_TBX)

def retain_top_k_by_abs(array: np.ndarray, k: int) -> np.ndarray:
    flat_array = array.flatten()
    if k <= 0:
        return np.zeros_like(array)
    k = min(k, flat_array.size)  # Ensure k does not exceed the number of elements
    top_k_indices = np.argpartition(np.abs(flat_array), -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(np.abs(flat_array[top_k_indices]))[::-1]]
    top_k_flat = np.zeros_like(flat_array)
    top_k_flat[top_k_indices] = flat_array[top_k_indices]

    return top_k_flat.reshape(array.shape)


print('')
print('Configuring explaining environment:')
wshapData_downsampled_np_flatten = background
train_wshap, est_wshap = train_test_split(wshapData_downsampled_np_flatten, test_size=0.3, random_state=randomseed)
Xtrain_wshap = train_wshap[:, :-1]
Xest_wshap = est_wshap[:, :-1]
ytrain_wshap = train_wshap[:, -1]
ytrain_wshap_onehot = tf.one_hot(ytrain_wshap, depth=2)
yest_wshap = est_wshap[:, -1]
yest_wshap_onehot = tf.one_hot(yest_wshap, depth=2)
ytest_wshap_onehot = tf.one_hot(ytest, depth=2)
Xtest_wshap = Xtest_downsampled_np_flatten
problem = 'classification'
ML_model = 'boosting'

class ModelWrapper:
    def __init__(self, model):
        self.model = model
    def __call__(self, data):
        prediction = self.model.predict_proba(data)[:, 1]
        return prediction
    def predict(self, data):
        return self.model.predict_proba(data)[:, 1]

task_model_callable = ModelWrapper(task_model)

# =====================================================================================================================
conditional_extension = weightedSHAP.generate_coalition_function(
    task_model_callable, Xtrain_wshap, Xest_wshap, problem, ML_model)
optimiser = TaylorPODA_engine.Taylor_PODA_optimiser(task_model_callable, Xbackground, maskModel=conditional_extension)
# =====================================================================================================================

print('')
c=2
n_sample = math.comb(downsample_length * downsample_length - 1, c-1)  # to cover all the c2 subsets
print('-------- Analyzing TaylorPODA_engine explanation --------')
optimised_attribution = optimiser.generate_optimised_attribution(
    input=X_TBX, options=16, dirichlet_scale=1, withMaskModel=1, rank=c, approx=True, n_sample=n_sample)
a_PODAc2 = np.round(optimised_attribution['Optimised attribution'], 6)
sum_PODAc2 = a_PODAc2.sum()
a_PODAc2_2Drebuild = np.expand_dims(np.array(a_PODAc2).reshape(downsample_length, downsample_length), -1)
a_PODAc2_top_k = retain_top_k_by_abs(a_PODAc2_2Drebuild, present_top_k)
a_PODAc2_upsampled = np.repeat(np.repeat(a_PODAc2_top_k, recover_factor, axis=0), recover_factor, axis=1)

original_image = Xtest[index]

vmin_podac2 = -max(abs(a_PODAc2_upsampled.min()), abs(a_PODAc2_upsampled.max()))
vmax_podac2 = max(abs(a_PODAc2_upsampled.min()), abs(a_PODAc2_upsampled.max()))

fig, axes = plt.subplots(1, 2, figsize=(5, 3.4))

def overlay_plot(ax, base_image, overlay_values, title, vmin, vmax, alpha=0.9, base_alpha=0.5):
    ax.imshow(base_image, cmap='gray', alpha=base_alpha)
    overlay = ax.imshow(overlay_values, cmap='coolwarm', alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=20)
    ax.axis("off")
    return overlay

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title("Original image", fontsize=20)
axes[0].axis("off")

podac2_title = "TaylorPODA"
overlay_plot(axes[1], original_image, a_PODAc2_upsampled, podac2_title, vmin_podac2, vmax_podac2)

plt.subplots_adjust(left=0, right=1, top=1, bottom=-0.08)
plt.show()
