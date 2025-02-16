# Orthopaedic Patient Classification

## Introduction

This project focuses on classifying orthopaedic patients into two categories: Normal (NO) and Abnormal (AB) based on six biomechanical features extracted from the pelvis and lumbar spine. The dataset contains 310 patient records, where six variables represent biomechanical characteristics, and one variable represents the class label.

## Data Cleaning

No null values were found in the dataset.

No duplicate values were present.

Outlier analysis using boxplots indicated the presence of outliers, especially in grade_of_spondylolisthesis and pelvic_tilt. However, since these are medical records, outliers were not removed to retain valuable insights.

## Exploratory Data Analysis (EDA)

5-Point Summary showed grade_of_spondylolisthesis had the highest standard deviation, ranging from -11.06 to 418.54.

Pair Plots revealed that pelvic_incidence had a strong positive relationship with all features except pelvic_radius, which showed a negative correlation.

Standardization was applied to the dataset to ensure all features contribute equally to distance calculations in modeling.

Label Encoding was performed on the class variable to convert categorical values into numerical labels.

## Unsupervised Clustering: K-Means

K-Means clustering was applied to discover hidden patterns in the unlabelled data.

The elbow method was used to determine the optimal number of clusters.

Based on the elbow plot, 3 clusters were selected.

Principal Component Analysis (PCA) with 2 components was used to visualize clusters.

Overlapping clusters suggested the possibility of an additional category beyond the two class labels provided.

## Supervised Classification: Support Vector Machine (SVM)

Support Vector Machine (SVM) was chosen for classification due to its efficiency in high-dimensional spaces.

The dataset was split into training (80%) and testing (20%) subsets.

The initial accuracy of the model was 85.48%.

Confusion Matrix was used to evaluate classification performance, computing precision, recall, and F1 score:

Accuracy: 85%

Precision: 76%

Recall: 72%

F1 Score: 74%

Hyperparameter Tuning with Grid Search improved accuracy to 87.1% by selecting the optimal RBF kernel.

## Conclusion

K-Means clustering suggested a possible third cluster, indicating some hidden structure in the data.

SVM classification effectively categorized patients with an 87.1% accuracy after hyperparameter tuning.

Using unsupervised learning (K-Means) before supervised classification (SVM) helped in feature importance analysis and data segmentation.

The non-linear RBF kernel proved to be more effective than a linear kernel for classification.

## Dependencies & Installation

To run the analysis, install the required Python libraries:

pip install numpy pandas seaborn scikit-learn matplotlib
