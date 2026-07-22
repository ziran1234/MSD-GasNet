# MSD-GasNet

Official PyTorch implementation of **MSD-GasNet** for GASF-based VOC classification using electronic-nose response signals.

This repository includes code for signal preprocessing, GASF image construction, model training and evaluation, traditional machine learning baselines, lightweight network comparisons, and result visualization.

The main scripts are:

- `graph_conversion.py`: time-series-to-image conversion;
- `model.py`: MSD-GasNet and related CNN models;
- `model2.py`: VGG-style models;
- `model3.py`: GhostNetV3, StarNet-S2, and FastViT-T8;
- `model_train_test.py`: deep learning experiments;
- `traditional_ml_baselines.py`: LR, RF, QDA, and KNN comparisons.

The raw datasets are available from the corresponding original publications and are not redistributed here.
