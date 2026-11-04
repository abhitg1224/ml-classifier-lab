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

The following results use the same 2,709 held-out observations for every model.
Values are rounded to six decimal places.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.919158 | 0.994717 | 0.930725 | 0.930049 | 0.930237 | 0.902314 |
| Decision Tree | 0.895533 | 0.946025 | 0.909421 | 0.910429 | 0.909784 | 0.873720 |
| KNN | 0.915467 | 0.983208 | 0.928993 | 0.925628 | 0.927027 | 0.897791 |
| Naive Bayes | 0.763012 | 0.966997 | 0.773610 | 0.768516 | 0.766895 | 0.714296 |
| Support Vector Machine | 0.921004 | 0.990615 | 0.934246 | 0.931703 | 0.932824 | 0.904468 |

| Model | Observations |
|---|---|
| Logistic Regression | Strong baseline with validation macro F1 of 0.936529 and the highest holdout AUC, 0.994717. Its holdout F1 is close to SVM, suggesting that linear boundaries capture much of the separation. |
| Decision Tree | Training accuracy of 1.000000 versus validation accuracy of 0.896345 indicates overfitting in the unrestricted tree. Holdout macro F1 is 0.909784. |
| KNN | Standardized five-neighbor classification achieves validation macro F1 of 0.934041 and holdout macro F1 of 0.927027, slightly below Logistic Regression and SVM. |
| Naive Bayes | Lowest validation macro F1, 0.763398, and holdout macro F1, 0.766895. This suggests that its Gaussian and conditional-independence assumptions are a weaker fit for these measurements. |
| Support Vector Machine | Highest training-only validation macro F1, 0.941061. It also achieves the highest holdout Accuracy, Precision, Recall, F1, and MCC, although Logistic Regression has higher AUC. |
| Overall winner: Support Vector Machine | Selected using training-only validation macro F1 before holdout evaluation. Holdout accuracy is 0.921004, macro F1 is 0.932824, and MCC is 0.904468. |
