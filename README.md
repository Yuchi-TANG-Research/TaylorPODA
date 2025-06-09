# TaylorPODA
This repository contains the implementation code associated with our research paper.  
The core functionalities are encapsulated within the `TaylorPODA_engine`.

- To reproduce the explanation results for a single input-output pair using **OCC-1**, **LIME**, **SHAP**, **WeightedSHAP**, and **TaylorPODA**,  
  please run `explain_concrete.py`. This enables a relatively fair comparison among different explanation methods.

- To generate an **approximated TaylorPODA explanation** for an MNIST image,  
  please run `MNIST_v5.py`.

The `weightedSHAP/` directory includes code adapted from the [WeightedSHAP] repository (https://github.com/ykwon0407/WeightedSHAP) that, while publicly available, does not include a formal open-source license. Permission to use and adapt this code was kindly granted by the original authors via email correspondence. We are grateful for their generosity in allowing its inclusion in this project.
