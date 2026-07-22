# MSD-GasNet

Official PyTorch implementation of **MSD-GasNet** for GASF-based volatile organic compound (VOC) classification using electronic-nose response signals.

This repository provides the main code used in the study, including signal preprocessing, GASF image construction, model implementation, training and evaluation, traditional machine learning baselines, lightweight network comparisons, and result visualization.

## Main Scripts

- `graph_conversion.py`: converts one-dimensional gas-response signals into image representations;
- `convert_gas_csv_to_gasf.py`: converts CSV-format gas-response data into GASF images;
- `model.py`: contains MSD-GasNet, its ablation variants, and related CNN models;
- `model2.py`: contains VGG-style classification and regression models;
- `model3.py`: wraps lightweight backbones, including GhostNetV3, StarNet-S2, and FastViT-T8;
- `model_train_test.py`: performs deep learning training, testing, and repeated evaluation;
- `traditional_ml_baselines.py`: implements Logistic Regression, Random Forest, Quadratic Discriminant Analysis, and k-Nearest Neighbors;
- `plot_five_fold_cv.py`: plots five-fold cross-validation curves;
- `plot_gas_response_curves.py`: plots representative gas-response curves;
- `draw_fig.py` and `csv_scatter_plot.py`: provide result visualization utilities.

## Example Outputs

The `examples/` folder contains selected representative original experimental outputs from Dataset 1 and Dataset 2.
