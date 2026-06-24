# TaylorPODA:  Taylor exPansion-Originated aDaptive Attribution
This repository contains the experimental implementation of the **TaylorPODA** method, proposed in the paper:

> **TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models**  

Parts of this project build upon code from [WeightedSHAP](https://github.com/ykwon0407/WeightedSHAP.git), used with the author’s permission.  

---

## Experiment Pipeline
Below are commands for running key components of the TaylorPODA method and the related experiments in the paper.

### A. Primary evaluation: quantitative performance under utility objectives: 
#### A.1 Model Training: 
```bash
python ml_classif.py
python ml_classif_relu.py
python ml_classif_xgb.py
python ml_regres.py
python ml_regres_relu.py
python ml_regres_xgb.py
```
#### A.2 Explanation Generarion & Quantitative Performance Analysis: 
(Please configure the path of the corresponding trained "task_model".)
```bash
python exp_cls.py
python exp_rgr.py
```

### B. Additional qualitative performance and examples: 
```bash
python MNIST_v5.py
```
