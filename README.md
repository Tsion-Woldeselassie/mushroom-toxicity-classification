# Mushroom Classification

## Summary
This repository implements machine learning models to classify mushrooms as edible or poisonous using categorical features from the [Kaggle Mushroom Classification Dataset](https://www.kaggle.com/uciml/mushroom-classification).

---

## Overview

### Definition of the Task
The goal of this project is to classify mushrooms as either **edible** (`e`) or **poisonous** (`p`) using their physical and biological features. The dataset contains 23 categorical features such as `odor`, `cap-shape`, `spore-print-color`, and more.

### Approach
This project formulates the problem as a binary classification task. Multiple machine learning models were developed and compared:
1. **Decision Tree:** For interpretability and handling categorical data.
2. **Random Forest:** For improved generalization and robustness against overfitting.
3. **Logistic Regression:** To test the performance of a linear model on the dataset.

### Performance Summary
- **Decision Tree Accuracy:** 100%  
- **Random Forest Accuracy:** 100%  
- **Logistic Regression Accuracy:** 99.83%  

The results demonstrate that the dataset is highly separable due to strong patterns in features like `odor` and `bruises`.

---

## Summary of Work Done

### Data

#### Dataset Details:
- **Type:** Categorical features in CSV format.
- **Size:** 8124 rows (instances) and 23 columns (features).
- **Class Distribution:**
  - Edible (`e`): 4208 instances (51.8%).
  - Poisonous (`p`): 3916 instances (48.2%).

#### Dataset Splits:
- **Training Set:** 70% of the data.
- **Validation Set:** 15% of the data.
- **Test Set:** 15% of the data.

### Preprocessing and Cleanup
1. **Missing Values:**
   - `stalk-root` had missing values in 2480 rows and was dropped due to limited predictive power.
   - `veil-type` had only one unique value and was also dropped.

2. **Feature Encoding:**
   - One-hot encoding was applied to all categorical features.
   - The target variable `class` was encoded as:
     - `e` (Edible): 0
     - `p` (Poisonous): 1

3. **Duplicate Handling:**
   - Removed duplicate rows after one-hot encoding to ensure data integrity.

---

### Data Visualization
#### Key Insights:
1. **Odor:**
   - `odor` is the most significant feature:
     - Mushrooms with `odor = n` (none) are mostly edible.
     - Mushrooms with `odor = f` (foul) or `odor = p` (pungent) are poisonous.
2. **Bruises:**
   - Bruised mushrooms (`bruises = t`) lean toward edibility, while no bruises suggest poison.
3. **Spore-Print Color:**
   - Certain spore-print colors (`spore-print-color = h` or `r`) correlate strongly with edibility or toxicity.

---

### Problem Formulation

#### Input / Output:
- **Input:** One-hot encoded categorical features.
- **Output:** Binary classification (`0` = Edible, `1` = Poisonous).

#### Models Used:
1. **Decision Tree:** 
   - Easy to interpret and naturally handles categorical features.
2. **Random Forest:**
   - Robust ensemble method to reduce overfitting.
3. **Logistic Regression:**
   - Linear model to validate if simpler relationships suffice.

---

### Training

#### Setup:
- **Environment:** Jupyter Notebook, Python (scikit-learn, pandas, matplotlib).
- **Hardware:** Standard laptop with CPU training.
- **Training Time:** All models trained in under 10 seconds.

#### Training Details:
- **Decision Tree:** Trained without depth limitations, achieving 100% accuracy.
- **Random Forest:** Used default hyperparameters, achieving 100% accuracy.
- **Logistic Regression:** Trained with `max_iter = 1000`, achieving 99.83% accuracy.

---

### Performance Comparison

#### Metrics:
- **Accuracy, Precision, Recall, F1-Score** were used to evaluate models.

#### Results:
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Decision Tree        | 100%     | 1.00      | 1.00   | 1.00     |
| Random Forest        | 100%     | 1.00      | 1.00   | 1.00     |
| Logistic Regression  | 99.83%   | 1.00      | 1.00   | 1.00     |

#### Visualizations:
1. **Feature Importance from Decision Tree:**
   - `odor` (63%) and `bruises` (16%) were the top predictors.
2. **Logistic Regression Coefficients:**
   - `spore-print-color_r` and `odor_c` had the strongest positive contributions.
## Overview of Files in Repository

| File/Folder           | Description                                                                      |
|-----------------------|----------------------------------------------------------------------------------|
| `mushrooms.csv`       | Dataset file with categorical features.                                          |
| `mushroom_classification_analysis.ipynb` | Jupyter Notebook with the full analysis, including preprocessing, modeling, and evaluation. |

## Link to Data
- The dataset used in this project is publicly available on Kaggle: [Mushroom Classification Dataset](https://www.kaggle.com/uciml/mushroom-classification).
### Conclusions

#### Key Takeaways:
1. Features like `odor` and `bruises` dominate the classification process, making the dataset highly separable.
2. Simpler models like Logistic Regression perform nearly as well as complex models, indicating that the dataset's patterns are straightforward.

#### Future Work:
1. Experiment with other models (e.g., Gradient Boosting, SVM).
2. Introduce noise or imbalances to test model robustness.

## Citations

1. **Dataset:**
   - [Kaggle Mushroom Classification Dataset](https://www.kaggle.com/uciml/mushroom-classification)
   - Source: UCI Machine Learning Repository

2. **Libraries:**
   - [pandas](https://pandas.pydata.org/): Data manipulation and analysis library.
   - [numpy](https://numpy.org/): Numerical computing library.
   - [matplotlib](https://matplotlib.org/): Data visualization library.
   - [seaborn](https://seaborn.pydata.org/): Statistical data visualization library.
   - [scikit-learn](https://scikit-learn.org/stable/): Machine learning library for Python.

3.. **Tools Used:**
   - [Jupyter Notebook](https://jupyter.org/): Interactive computing environment.
   - [Python](https://www.python.org/): Programming language used for analysis.

---

