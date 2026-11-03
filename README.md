# Dry Bean Classification

## a. Problem statement

Classify dry beans into seven varieties using 16 numerical shape measurements.
Compare Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and Support
Vector Machine using the same dataset and held-out test partition.

## b. Dataset description

The UCI Dry Bean dataset contains 13,611 observations, 16 numerical input features,
and the target column `Class`. It has no missing values.

- Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- DOI: https://doi.org/10.24432/C50S4B
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.
- Features: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
  Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness,
  Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, and ShapeFactor4.

Exact duplicate observations are removed before a stratified 80:20 train/test
split with random seed 42. Identical feature vectors with conflicting labels are
rejected. Removing 68 duplicates leaves 13,543 unique observations: 10,834
training rows and 2,709 test rows, with no shared feature vectors.

## c. Github Repository Link

https://github.com/abhitg1224/ml-classifier-lab

## d. Models used

The five classifiers are Logistic Regression, Decision Tree, KNN with five
neighbors, Gaussian Naive Bayes, and an RBF Support Vector Machine.

Logistic Regression and KNN use standardization fitted only on training data.
SVM uses sigmoid probability calibration with standardization inside each
calibration fold. All models use the same three stratified training-only
validation folds; mean validation macro F1 determines the selected model.

Evaluation uses Accuracy, AUC, Precision, Recall, F1, and MCC. Precision, Recall,
and F1 are macro-averaged over the seven classes. AUC is macro one-vs-rest ROC AUC
from class-aligned probabilities and is unavailable if the test data omits a
class. MCC uses the multiclass definition.
