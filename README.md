# GasRecognition

GasRecognition is an electronic-nose gas recognition project. It converts gas response CSV signals into image representations such as GASF, RP, MTF, CWT, and STFT, then uses deep learning and traditional machine learning methods for gas classification, concentration regression, and visualization.

`model.py` contains the main custom CNN models, including multi-scale, attention, residual, ablation, plain CNN, and AlexNet-style variants. `model2.py` provides compact VGG19-style models for classification and regression. `model3.py` wraps lightweight `timm` backbones, including GhostNetV3, StarNet-S2, and FastViT-T8.

The main experiment scripts are `graph_conversion.py` for signal-to-image conversion, `model_train_test.py` for deep learning experiments, and `traditional_ml_baselines.py` for machine learning baseline comparisons.

Experiment outputs, including metric CSV files, loss and accuracy curves, confusion matrices, and clustering visualizations, are saved in folders such as `train_test_results/`, `confusion_matrix/`, `cluster/`, and `traditional_ml_results/`.
