# Survival Analysis with scikit-survival: A Tutorial

## Introduction

Survival analysis is a branch of statistics for analyzing the expected duration of time until one or more events happen, such as death in biological organisms and failure in mechanical systems. This tutorial demonstrates how to perform survival analysis using Python, primarily with the `scikit-survival` library, on the Veteran's Administration Lung Cancer dataset.

**scikit-survival** is an open-source Python library for survival analysis built on top of scikit-learn. It provides a rich set of models, metrics, and utility functions for tackling survival analysis problems.

The **Veteran's Administration Lung Cancer dataset** is a classic dataset used for survival analysis examples. It contains data from a study on male patients with advanced, inoperable lung cancer, comparing standard and test chemotherapy treatments.

This tutorial will cover:
- Data loading and preprocessing.
- Exploratory data analysis with Kaplan-Meier curves.
- Fitting and interpreting several survival models:
    - Cox Proportional Hazards model.
    - Cox PH model with L1 (Lasso) regularization for feature selection.
    - Random Survival Forests.
    - Gradient Boosted Survival Analysis.
- Comparing these models based on the concordance index.

## Setup

### Required Libraries
You will need the following Python libraries:
- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `matplotlib` for plotting (though actual plot generation is placeholder in this text-based tutorial).
- `scikit-survival` for survival analysis modeling.

### Installation
You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib scikit-survival
```

## 1. Data Loading and Preprocessing

The first step is to load our dataset and prepare it for analysis. The Veteran's dataset has features like treatment type, cell type of the tumor, prior therapy, age, diagnostic scores, etc.

### Dataset Columns (from `colnames.csv`)
- `Treatment`: Type of treatment (1=standard, 2=test).
- `Celltype`: Type of cancer cell (1=squamous, 2=smallcell, 3=adeno, 4=large).
- `Survival_days`: Time to event or censoring in days.
- `Status`: Event indicator (1=death, 0=censored).
- `Karnofsky_Score`: A measure of performance status (0-100, higher is better).
- `Months_Diagnosis`: Months from diagnosis to study entry.
- `Age`: Age in years.
- `Prior_Therapy`: Prior therapy (0=no, 10=yes).

### Preprocessing Script (`preprocess_data.py` logic)
The following Python code loads the data, assigns column names, creates the necessary structured array for `scikit-survival` (event indicator and duration), and performs one-hot encoding for categorical features.

```python
import pandas as pd
from sksurv.util import Surv

def load_and_preprocess_data():
    # 1. Load data
    data_path = 'dataset/vet-admin.csv' # Assuming data is in a 'dataset' subfolder
    df = pd.read_csv(data_path, header=None)

    # 2. Load column names
    colnames_path = 'dataset/colnames.csv'
    try:
        colnames_df = pd.read_csv(colnames_path, header=0)
        colnames = colnames_df.columns.tolist()
    except Exception as e:
        with open(colnames_path, 'r') as f:
            line = f.readline().strip()
            colnames = [name.strip() for name in line.split(',')]
    df.columns = colnames

    # 4. Prepare target variable for scikit-survival
    # 'Status' from colnames.csv is the event status (1=event, 0=censored)
    # 'Survival_days' from colnames.csv is the time.
    df['Status'] = df['Status'].apply(lambda x: x == 1)
    # df['Survival_days'] is already numeric.

    # Store this version of df for EDA, it has original categories and processed Status/Survival_days
    df_for_eda = df.copy()

    # 5. Feature Engineering (One-Hot Encoding) for X
    df_for_ohe = df.copy()
    categorical_cols = ['Treatment', 'Celltype', 'Prior_Therapy']
    df_for_ohe['Celltype'] = df_for_ohe['Celltype'].astype(str) # Ensure Celltype is treated as categorical

    df_encoded = pd.get_dummies(df_for_ohe, columns=categorical_cols, drop_first=True)

    cols_to_drop_for_X = ['Status', 'Survival_days']
    X = df_encoded.drop(columns=cols_to_drop_for_X)

    # Create the structured array for survival analysis using df_for_eda
    y_structured = Surv.from_dataframe(event='Status', time='Survival_days', data=df_for_eda)

    return X, y_structured, df_for_eda

# Example of running the preprocessing and showing outputs:
if __name__ == '__main__': # Simulating direct run for README
    X_processed, y_sksurv, df_original_eda = load_and_preprocess_data()

    print("\n--- Shape of Feature Matrix X ---")
    print(X_processed.shape)
    # Expected Output: (137, 8)

    print("\n--- First Few Rows of Feature Matrix X ---")
    print(X_processed.head())
    # Expected Output (column names might vary slightly based on get_dummies version/behavior):
    #    Karnofsky_Score  Months_Diagnosis  Age  Treatment_2  Celltype_2  Celltype_3  Celltype_4  Prior_Therapy_10
    # 0               60                 7   69        False       False       False       False             False
    # 1               70                 5   64        False       False       False       False              True
    # 2               60                 3   38        False       False       False       False             False
    # 3               60                 9   63        False       False       False       False              True
    # 4               70                11   65        False       False       False       False              True

    print("\n--- First Few Elements of Target Variable y (event, duration) ---")
    print(y_sksurv[:5])
    # Expected Output:
    # [( True,  72.) ( True, 411.) ( True, 228.) ( True, 126.) ( True, 118.)]

    print("\n--- df_original_eda head (for EDA grouping) ---")
    print(df_original_eda.head())
    # Expected Output (shows original categorical columns before OHE):
    #    Treatment  Celltype  Survival_days  Status  Karnofsky_Score  Months_Diagnosis  Age  Prior_Therapy
    # 0          1         1             72    True               60                 7   69              0
    # 1          1         1            411    True               70                 5   64             10
    # 2          1         1            228    True               60                 3   38              0
    # 3          1         1            126    True               60                 9   63             10
    # 4          1         1            118    True               60                11   65             10

```

**Explanation:**
- The `Status` column is converted to boolean (True for event, False for censored).
- `Survival_days` provides the time to event or censoring.
- `scikit-survival` requires the target variable `y` to be a structured NumPy array, where each element is a tuple `(event_indicator, time_to_event)`. `Surv.from_dataframe` helps create this.
- Categorical features (`Treatment`, `Celltype`, `Prior_Therapy`) are one-hot encoded using `pd.get_dummies`. `drop_first=True` is used to avoid multicollinearity by removing one category per feature. `Celltype` is converted to string type before encoding to ensure its numerical values are treated as distinct categories.
- The function returns `X_ohe` (one-hot encoded features), `y_sksurv` (structured target array), and `df_original_eda` (DataFrame with original categorical values, useful for stratified Kaplan-Meier).

## 2. Exploratory Data Analysis: Kaplan-Meier Curves

The Kaplan-Meier estimator is a non-parametric statistic used to estimate the survival function from lifetime data. It's a good first step to visualize survival probabilities.

### Kaplan-Meier Analysis Script (`kaplan_meier_analysis.py` logic)
```python
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival # For logrank_test
# Assuming load_and_preprocess_data is available from preprocess_data.py

# X_ohe, y_sksurv, data_df_eda = load_and_preprocess_data()
# (This line would be here to load data in a real script)

# For README: Simulating loaded data for context
# y_sksurv would be the structured array from preprocessing.
# data_df_eda would be the DataFrame with original 'Treatment' and 'Celltype'.

# --- Overall Kaplan-Meier Survival Analysis ---
# time, survival_prob = kaplan_meier_estimator(y_sksurv["Status"], y_sksurv["Survival_days"])
# plt.step(time, survival_prob, where="post", label="Overall Survival")
# plt.title("Overall Kaplan-Meier Survival Curve")
# plt.ylabel("Survival Probability")
# plt.xlabel("Time (Days)")
# plt.grid(True)
# plt.show() # Code to generate and display the plot would go here.
print("Placeholder: Overall Kaplan-Meier plot would be generated here.")

# --- Kaplan-Meier Analysis by Treatment Group ---
treatment_col = 'Treatment'
# for group_val in sorted(data_df_eda[treatment_col].unique()):
#     mask = (data_df_eda[treatment_col] == group_val)
#     time_group, survival_prob_group = kaplan_meier_estimator(
#         y_sksurv["Status"][mask], y_sksurv["Survival_days"][mask]
#     )
#     plt.step(time_group, survival_prob_group, where="post", label=f"{treatment_col} {group_val}")
# plt.title("Kaplan-Meier Survival Curve by Treatment Group")
# plt.legend()
# plt.show() # Code to generate and display the plot would go here.
print(f"Placeholder: Kaplan-Meier plots for {treatment_col} groups would be generated here.")

# treatment_groups = data_df_eda[treatment_col]
# chisq_treat, p_val_treat, _, _ = compare_survival(y_sksurv, treatment_groups, return_stats=True)
# print(f"Log-rank test for {treatment_col}: Chi-squared Statistic={chisq_treat:.4f}, p-value={p_val_treat:.4f}")
# Expected Output: Log-rank test for Treatment: Chi-squared Statistic=0.0082, p-value=0.9277

# --- Kaplan-Meier Analysis by Cell Type ---
celltype_col = 'Celltype'
# for group_val in sorted(data_df_eda[celltype_col].unique()):
#     mask = (data_df_eda[celltype_col] == group_val)
#     time_group, survival_prob_group = kaplan_meier_estimator(
#         y_sksurv["Status"][mask], y_sksurv["Survival_days"][mask]
#     )
#     plt.step(time_group, survival_prob_group, where="post", label=f"{celltype_col} {group_val}")
# plt.title("Kaplan-Meier Survival Curve by Cell Type")
# plt.legend()
# plt.show() # Code to generate and display the plot would go here.
print(f"Placeholder: Kaplan-Meier plots for {celltype_col} groups would be generated here.")

# celltype_groups = data_df_eda[celltype_col]
# chisq_cell, p_val_cell, _, _ = compare_survival(y_sksurv, celltype_groups, return_stats=True)
# print(f"Log-rank test for {celltype_col}: Chi-squared Statistic={chisq_cell:.4f}, p-value={p_val_cell:.4f}")
# Expected Output: Log-rank test for Celltype: Chi-squared Statistic=25.4037, p-value=0.0000
```

**Interpretation:**
- **Overall Survival**: The plot would show the estimated probability of survival over time for all patients in the study.
- **Survival by Treatment**:
    - The plots would compare survival curves for patients receiving standard vs. test treatment.
    - The log-rank test for 'Treatment' yielded a p-value of approximately 0.9277. Since this p-value is much greater than 0.05, we do not have statistically significant evidence to conclude that there is a difference in survival times between the two treatment groups.
- **Survival by Cell Type**:
    - The plots would compare survival curves for patients based on their tumor cell type.
    - The log-rank test for 'Celltype' yielded a p-value of approximately 0.0000. This very small p-value indicates a statistically significant difference in survival experiences among the different cell types. Some cell types are associated with poorer prognosis than others.

## 3. Cox Proportional Hazards Model

The Cox Proportional Hazards (PH) model is a semi-parametric model that describes the relationship between the event incidence, as expressed by the hazard function, and a set of covariates.

### Cox PH Analysis Script (`cox_ph_analysis.py` logic)
```python
import pandas as pd
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
# Assuming load_and_preprocess_data and median_survival_times (or fallback) are available

# X_ohe, y_sksurv, _ = load_and_preprocess_data()
# (This line would be here to load data in a real script)

# model_cox = CoxPHSurvivalAnalysis(alpha=0.1) # Using L2 regularization (alpha is penalty strength)
# model_cox.fit(X_ohe, y_sksurv)

# coefficients = model_cox.coef_
# feature_names = X_ohe.columns.tolist()
# coef_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Coefficient': coefficients,
#     'Hazard Ratio (HR)': np.exp(coefficients)
# }).sort_values(by='Hazard Ratio (HR)', ascending=False)
# print(coef_df)
# Expected Output (example, actual values depend on exact X_ohe columns):
#             Feature  Coefficient  Hazard Ratio (HR)
# 5        Celltype_3     1.172325           3.229492
# 4        Celltype_2     0.842611           2.322424
# 6        Celltype_4     0.388052           1.474106
# 3       Treatment_2     0.286619           1.331916
# 7  Prior_Therapy_10     0.070209           1.072733
# 1  Months_Diagnosis    -0.000051           0.999949
# 2               Age    -0.008482           0.991554
# 0   Karnofsky_Score    -0.032617           0.967909

# c_index_cox = model_cox.score(X_ohe, y_sksurv)
# print(f"Concordance Index (C-index) on training data: {c_index_cox:.4f}")
# Expected Output: Concordance Index (C-index) on training data: 0.7363

# X_sample = X_ohe.head(3)
# surv_funcs_cox = model_cox.predict_survival_function(X_sample)
# print("Placeholder: Predicted survival functions for Cox PH model would be plotted here.")
# median_preds_cox = median_survival_times(surv_funcs_cox) # Using our utility
# print(f"Predicted median survival times for sample patients (Cox PH): {median_preds_cox}")
# Expected Output (example): Predicted median survival times for sample patients (Cox PH): [162. 201. 132.]
```

**Interpretation:**
- **Coefficients and Hazard Ratios (HR)**:
    - A positive coefficient (HR > 1) implies that an increase in the feature's value is associated with an increased hazard of the event (e.g., death). For example, `Celltype_3` (Adeno) having an HR of ~3.23 suggests it's associated with a higher risk compared to the baseline cell type (Squamous, since `Celltype_1` was dropped).
    - A negative coefficient (HR < 1) implies a decreased hazard. `Karnofsky_Score` having an HR of ~0.97 suggests that higher scores (better patient performance) are associated with a lower risk of death.
- **Concordance Index (C-index)**: The C-index was approximately 0.7363. This value (often considered fair to good) indicates the model's ability to correctly rank pairs of subjects by survival time. A C-index of 0.5 is random, while 1.0 is perfect.
- **Prediction**: The model can predict survival functions for new patients, showing their estimated survival probability over time. Median survival times can also be derived.

## 4. Feature Selection with Cox PH (L1 Regularization - Lasso)

Regularization can help prevent overfitting and perform feature selection by shrinking some coefficients to zero. Lasso (L1 penalty) is particularly useful for this.

### Cox PH Lasso Analysis Script (`cox_ph_lasso_analysis.py` logic)
```python
from sksurv.linear_model import CoxnetSurvivalAnalysis # Correct model for L1/Lasso

# X_ohe, y_sksurv, _ = load_and_preprocess_data()
# (This line would be here to load data)

chosen_alpha_lasso = 0.05 # Example alpha value
# model_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[chosen_alpha_lasso], fit_baseline_model=True)
# model_lasso.fit(X_ohe, y_sksurv)

# coefficients_lasso = model_lasso.coef_[:, 0] # Coeffs for the chosen alpha
# feature_names_lasso = X_ohe.columns.tolist()
# lasso_coef_df = pd.DataFrame({
#     'Feature': feature_names_lasso,
#     'Coefficient': coefficients_lasso,
#     'Hazard Ratio (HR)': np.exp(coefficients_lasso)
# })
# selected_features_df = lasso_coef_df[lasso_coef_df['Coefficient'] != 0]
# print(selected_features_df.sort_values(by='Hazard Ratio (HR)', ascending=False))
# Expected Output (example for alpha=0.05):
#             Feature  Coefficient  Hazard Ratio (HR)
# 5        Celltype_3     0.385414           1.470223
# 4        Celltype_2     0.111399           1.117840
# 1  Months_Diagnosis     0.001469           1.001470
# 2               Age    -0.002760           0.997244
# 0   Karnofsky_Score    -0.032749           0.967781

# print(f"Features selected by Lasso: {len(selected_features_df)} out of {X_ohe.shape[1]}")
# Expected Output: Features selected by Lasso: 5 out of 8

# c_index_lasso = model_lasso.score(X_ohe, y_sksurv)
# print(f"C-index (Lasso, alpha={chosen_alpha_lasso}): {c_index_lasso:.4f}")
# Expected Output: C-index (Lasso, alpha=0.05): 0.7293
# print(f"Reference C-index (L2-regularized CoxPH): 0.7363") # From previous script
```

**Interpretation:**
- `CoxnetSurvivalAnalysis` is used for L1/Lasso regularization (`l1_ratio=1.0`).
- With `alpha=0.05`, Lasso selected 5 out of 8 features. Features like `Treatment_2`, `Celltype_4`, and `Prior_Therapy_10` had their coefficients shrunk to zero, effectively removing them from the model.
- The C-index for this Lasso model (0.7293) is slightly lower than the L2-regularized Cox PH model (0.7363). This illustrates a common trade-off: Lasso achieves sparsity (simpler model) potentially at the cost of a slight decrease in predictive performance on the training set if the excluded features had some predictive value.
- **Alpha Tuning**: The `alpha` parameter (regularization strength) is a hyperparameter that requires careful tuning, often using cross-validation, to find the optimal balance between model complexity and predictive accuracy.

## 5. Random Survival Forests (RSF)

Random Survival Forests are an ensemble method that extends random forests to survival data. They are non-parametric and can capture complex relationships between features and survival.

### RSF Analysis Script (`rsf_analysis.py` logic)
```python
from sksurv.ensemble import RandomSurvivalForest
# from sksurv.inspection import permutation_importance # Attempted import

# X_ohe, y_sksurv, _ = load_and_preprocess_data()
# (This line would be here to load data)

# rsf_params = {
#     'n_estimators': 100, 'min_samples_split': 10,
#     'min_samples_leaf': 15, 'random_state': 42, 'n_jobs': -1
# }
# rsf_model = RandomSurvivalForest(**rsf_params)
# rsf_model.fit(X_ohe, y_sksurv)

# c_index_rsf_train = rsf_model.score(X_ohe, y_sksurv)
# print(f"C-index on training data (RSF): {c_index_rsf_train:.4f}")
# Expected Output: C-index on training data (RSF): 0.7561

# print("\n--- Variable Importance (RSF) ---")
# try:
#     from sksurv.inspection import permutation_importance
#     perm_importance_result = permutation_importance(
#         rsf_model, X_ohe, y_sksurv, n_repeats=15, random_state=42, n_jobs=-1
#     )
#     # Process and print perm_importance_result...
#     print("Placeholder: Permutation importances would be shown here if 'sksurv.inspection' was available.")
# except ModuleNotFoundError:
#     print("\nModule 'sksurv.inspection' not found. Cannot calculate permutation importance.")
#     print("RSF `feature_importances_` attribute is also not implemented in this version.")
# Expected Output (if module missing, as in tutorial run):
# Module 'sksurv.inspection' not found. Cannot calculate permutation importance.
# RSF `feature_importances_` attribute is also not implemented in this version.

# X_sample_rsf = X_ohe.head(3)
# surv_funcs_rsf = rsf_model.predict_survival_function(X_sample_rsf)
# print("Placeholder: Predicted survival functions for RSF model would be plotted here.")
# median_preds_rsf = median_survival_times(surv_funcs_rsf) # Using our utility
# print(f"Predicted median survival times for sample patients (RSF): {median_preds_rsf}")
# Expected Output (example): Predicted median survival times for sample patients (RSF): [87. 112. 90.]
```

**Interpretation:**
- The RSF model achieved a C-index of approximately 0.7561 on the training data. This is higher than the Cox PH models, suggesting it might be capturing more complex patterns or potentially overfitting to the training data (cross-validation would be needed to assess this).
- **Feature Importance**:
    - The `feature_importances_` attribute (impurity-based) is not implemented for `RandomSurvivalForest` in `scikit-survival`.
    - The preferred method is permutation importance using `sksurv.inspection.permutation_importance`. However, during the execution of this tutorial's scripts, the `sksurv.inspection` module was not found, possibly due to the specific version or installation state of `scikit-survival` in the environment. Thus, feature importances for RSF could not be displayed. If available, permutation importance assesses a feature's importance by measuring the decrease in model performance when that feature's values are randomly shuffled.
- **Prediction**: RSF can also predict individual survival functions and median survival times.

## 6. Gradient Boosted Survival Analysis (GBSA)

Gradient Boosting is another powerful ensemble technique that builds models sequentially, with each new model correcting errors made by previous ones. `GradientBoostingSurvivalAnalysis` applies this to survival data, often using a Cox-like loss function.

### GBSA Script (`gbsa_analysis.py` logic)
```python
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

# X_ohe, y_sksurv, _ = load_and_preprocess_data()
# (This line would be here to load data)

# gbsa_params = {
#     'n_estimators': 100, 'learning_rate': 0.1,
#     'max_depth': 3, 'random_state': 42
# }
# gbsa_model = GradientBoostingSurvivalAnalysis(**gbsa_params)
# gbsa_model.fit(X_ohe, y_sksurv)

# c_index_gbsa_train = gbsa_model.score(X_ohe, y_sksurv)
# print(f"C-index on training data (GBSA): {c_index_gbsa_train:.4f}")
# Expected Output: C-index on training data (GBSA): 0.8613

# if hasattr(gbsa_model, 'feature_importances_'):
#     feature_importances_gbsa = gbsa_model.feature_importances_
#     feature_names_gbsa = X_ohe.columns.tolist()
#     importance_df_gbsa = pd.DataFrame({
#         'Feature': feature_names_gbsa,
#         'Importance': feature_importances_gbsa
#     }).sort_values(by='Importance', ascending=False)
#     print(importance_df_gbsa)
# Expected Output:
#             Feature  Importance
# 0   Karnofsky_Score    0.378849
# 2               Age    0.220423
# 1  Months_Diagnosis    0.167217
# 5        Celltype_3    0.096994
# 4        Celltype_2    0.068054
# 3       Treatment_2    0.037552
# 7  Prior_Therapy_10    0.026462
# 6        Celltype_4    0.004449

# X_sample_gbsa = X_ohe.head(3)
# surv_funcs_gbsa = gbsa_model.predict_survival_function(X_sample_gbsa)
# print("Placeholder: Predicted survival functions for GBSA model would be plotted here.")
# median_preds_gbsa = median_survival_times(surv_funcs_gbsa) # Using our utility
# print(f"Predicted median survival times for sample patients (GBSA): {median_preds_gbsa}")
# Expected Output (example): Predicted median survival times for sample patients (GBSA): [ 99. 177. 118.]
```

**Interpretation:**
- The GBSA model achieved a C-index of approximately 0.8613 on the training data. This is the highest training C-index among the models tested, suggesting a strong fit to the training data. As with RSF, its generalization performance should be validated using cross-validation or a test set.
- **Feature Importance**: `GradientBoostingSurvivalAnalysis` provides a `feature_importances_` attribute (typically based on impurity or loss reduction).
    - `Karnofsky_Score` was found to be the most important feature, followed by `Age` and `Months_Diagnosis`. This aligns with clinical intuition that patient performance status and age are significant factors.
- **Prediction**: GBSA, like other models, can predict individual survival functions and median survival times.

## 7. Comparing Models

Here's a summary of the Concordance Index (C-index) values obtained on the **training data** for the different models explored in this tutorial:

| Model                       | C-index (Training) | Notes                                     |
|-----------------------------|--------------------|-------------------------------------------|
| Cox PH (L2, alpha=0.1)      | 0.7363             | Standard regularized linear model         |
| Cox PH (L1/Lasso, alpha=0.05)| 0.7293             | 5 out of 8 features selected              |
| Random Survival Forest (RSF)| 0.7561             | Ensemble tree model                       |
| Gradient Boosting (GBSA)    | 0.8613             | Ensemble boosting model                   |

**Note**: These C-index values are for the training data only and do not reflect performance on unseen data. GBSA shows the highest C-index on training data, but this might not always translate to the best performance on a test set due to potential overfitting. Proper hyperparameter tuning and cross-validation are essential for robust model comparison and selection.

## 8. Conclusion

This tutorial covered key aspects of survival analysis using `scikit-survival` on the Veteran's lung cancer dataset. We performed:
- Data preprocessing, including one-hot encoding and creation of structured survival arrays.
- Exploratory analysis using Kaplan-Meier curves and log-rank tests, which highlighted 'Celltype' as a significant factor in survival.
- Application and interpretation of several survival models:
    - Cox Proportional Hazards model (with L2 regularization).
    - Cox PH with Lasso (L1) regularization for feature selection.
    - Random Survival Forests.
    - Gradient Boosting Survival Analysis.
- We found that 'Celltype' and 'Karnofsky_Score' are generally important predictors. The GBSA model achieved the highest C-index on the training data, suggesting its potential for capturing complex relationships, though this requires validation.

## 9. Further Steps

This tutorial provides a foundation. Further exploration could include:
- **Hyperparameter Tuning**: Use cross-validation (e.g., `sklearn.model_selection.GridSearchCV` adapted for survival data or specific utilities in `scikit-survival`) to find optimal hyperparameters for each model.
- **Model Validation**: Evaluate models on a separate test set or using robust cross-validation schemes to get a better estimate of generalization performance.
- **Other Models**: Explore other models in `scikit-survival` like Support Vector Machines for survival (Survival SVM) or more advanced ensemble techniques.
- **Time-Dependent Covariates**: If applicable, incorporate covariates that change over time.
- **Competing Risks**: Analyze scenarios where individuals are at risk of multiple types of events.
- **Deep Learning**: Investigate neural network-based approaches for survival analysis (e.g., DeepSurv, Neural Survival Networks).

## 10. Dataset Citation

The Veteran's Administration Lung Cancer Study dataset is a classic dataset. One common citation is:
Kalbfleisch, J. D. and Prentice, R. L. (1980) The Statistical Analysis of Failure Time Data. New York: Wiley.

(The data itself is often attributed to a study by the Veteran's Administration, but the Kalbfleisch and Prentice book is a primary reference for its use in survival analysis examples.)

---
This concludes the tutorial. Remember that the specific results and interpretations can vary based on data preprocessing choices, hyperparameter settings, and the version of libraries used.
