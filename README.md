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
rejected.

## c. Github Repository Link

https://github.com/abhitg1224/ml-classifier-lab

## d. Models used

Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and Support Vector
Machine will be compared using Accuracy, AUC, Precision, Recall, F1, and MCC.
