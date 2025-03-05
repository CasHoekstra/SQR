#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Step 1: Import packages
import numpy as np
import scipy
import sympy as sp
import torch
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_pinball_loss
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import friedmanchisquare, wilcoxon
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import pmlb
from pmlb import fetch_data, regression_dataset_names
import random
import pandas as pd

# Step 2: Import pysr AFTER Julia dependencies are configured
from pysr import PySRRegressor


# In[20]:


def get_feature_type(data_column):
  """
  This function determines the feature type (categorical or binary) for a NumPy array column.

  **Note:** This function assumes the data doesn't contain missing values.
          If your data might have missing values, you'll need to handle them
          before using this function (e.g., impute missing values or remove rows).

  Args:
      data_column: A NumPy array representing the data column.

  Returns:
      A string indicating the feature type: "categorical" or "binary".
  """

  # Check for distinct values and data type
  unique_values = np.unique(data_column)
  num_unique_values = len(unique_values)

  # Categorical data has a limited number of distinct values (adjust threshold as needed)
  if num_unique_values <= 10:  # Adjust threshold based on your data and analysis goals
    return "categorical"

  # Binary data has only two distinct values
  if num_unique_values == 2:
    return "binary"

  # If there are more than 10 distinct values and not binary, assume numerical
  # (This might need further refinement depending on your domain knowledge)
  return "numerical"  # Consider a different label for non-categoric

def get_categorical_features(X):
  """
  This function identifies the indices of categorical features in a NumPy array representing a dataset.

  **Note:** This function assumes the data doesn't contain missing values.
          If your data might have missing values, you'll need to handle them
          before using this function (e.g., impute missing values or remove rows).

  Args:
      X: A 2D NumPy array representing the dataset (n_samples x n_features).

  Returns:
      A list of integers representing the indices of categorical features in X.
  """

  categorical_features = []
  for i, col in enumerate(X.T):  # Enumerate to get column index (i)
    unique_values = np.unique(col)
    num_unique_values = len(unique_values)
    data_type = col.dtype

    # Categorical data has a limited number of distinct values (adjust threshold as needed)
    if num_unique_values <= 10:  # Adjust threshold based on your data and analysis goals
      categorical_features.append(i)

  return categorical_features


def create_dummy_variables(X, categorical_features):
    """
    This function creates dummy variables for categorical features in a dataset.

    Args:
        X: A 2D NumPy array representing the dataset (n_samples x n_features).
        categorical_features: A list of integers representing the indices of categorical features in X.

    Returns:
        A 2D NumPy array representing the dataset with dummy variables for categorical features.
    """
    if categorical_features == []:
        return X
    else:
        # Select categorical features from the data
        X_categorical = X[:, categorical_features]

        # Create one-hot encoder
        encoder = OneHotEncoder(sparse_output=False)

        # Fit the encoder on the categorical features
        encoder.fit(X_categorical)

        # Transform the categorical features into dummy variables
        X_dummy_categorical = encoder.transform(X_categorical)

        # Get the original non-categorical features (assuming they are numerical)
        X_numerical = np.delete(X, categorical_features, axis=1)  # Delete categorical feature columns

        # Combine the dummy variables and numerical features
        X_with_dummies = np.concatenate([X_numerical, X_dummy_categorical], axis=1)

        return X_with_dummies


# In[21]:


regression_dataset_namestry = regression_dataset_names
#regression_dataset_namestry = regression_dataset_names[2:3]

print(regression_dataset_namestry)


# SQR BENCHMARK **90TH** QUANTILE

# In[ ]:


# Set seed for reproducibility
SEED = 42  # Change as needed

# Set NumPy seed
np.random.seed(SEED)

# Set Python random seed
random.seed(SEED)

# Set PyTorch seed (if using)
torch.manual_seed(SEED)

# Set Optuna seed
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce logging clutter
optuna_seed = SEED

# Global quantile setting (EXCEPT FOR PYSR, THIS NEEDS MANUAL ADJUSTMENT)
QUANTILE = 0.9 #(CHANGE PYSR QUANTILE MANUALLY)

# Function to calculate pinball loss
def pinball_loss(y_true, y_pred, tau=QUANTILE):
    residuals = y_true - y_pred
    loss = np.where(residuals >= 0, tau * residuals, (1 - tau) * -residuals)
    return np.mean(loss)

# Function to calculate normalized pinball loss using the global dataset range
def normalized_pinball_loss(y_true, y_pred, global_min, global_max, tau=QUANTILE):
    range_y = global_max - global_min
    loss = pinball_loss(y_true, y_pred, tau)
    return loss / range_y if range_y != 0 else 0  # Avoid division by zero

# Function to calculate absolute coverage error
def absolute_coverage_error(y_true, y_pred, tau=QUANTILE):
    coverage = np.mean(y_pred >= y_true)
    return np.abs(coverage - tau)

# Function to calculate expression complexity
def calculate_expression_complexity(expression, complexity_of_operators):
    try:
        expr = sp.sympify(expression)
    except sp.SympifyError:
        raise ValueError("Invalid expression")

    complexity = 0
    for atom in sp.preorder_traversal(expr):
        if isinstance(atom, sp.Symbol):  # Variables (e.g., x1, x2)
            complexity += 1
        elif isinstance(atom, (int, float, sp.Integer, sp.Float)):  # Constants
            complexity += 1
        elif atom in complexity_of_operators:  # Operators
            complexity += complexity_of_operators[atom]
    return complexity

# LightGBM objective function for Optuna
def objective_lgb(trial, train_X, train_y, val_X, val_y):
    params = {
        'objective': 'quantile',
        'alpha': QUANTILE,
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': SEED,
        'bagging_seed': SEED,
        'feature_fraction_seed': SEED,
        'data_random_seed': SEED,
    }

    if params['min_child_samples'] >= params['num_leaves']:
        raise optuna.exceptions.TrialPruned()

    model = lgb.LGBMRegressor(**params)
    model.fit(train_X, train_y)
    y_pred = model.predict(val_X)
    return pinball_loss(val_y, y_pred, tau=QUANTILE)

# Quantile regression objective function for Optuna
def objective_linear(trial, train_X, train_y, val_X, val_y, tau=QUANTILE):
    max_iter = trial.suggest_int('max_iter', 1000, 5000)
    model = QuantReg(train_y, train_X)
    results = model.fit(q=tau, max_iter=max_iter)
    y_pred = results.predict(val_X)
    return pinball_loss(val_y, y_pred, tau)

# Quantile Decision Tree Regressor
class QuantileDecisionTreeRegressor:
    def __init__(self, quantile=QUANTILE, min_samples_leaf=5, random_state=SEED):
        self.quantile = quantile
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=random_state)

    def fit(self, X, y):
        self.tree.fit(X, y)
        self._add_quantile_info(X, y)

    def _add_quantile_info(self, X, y):
        leaf_indices = self.tree.apply(X)
        unique_leaves = np.unique(leaf_indices)
        self.quantile_values = {}
        for leaf in unique_leaves:
            leaf_y = y[leaf_indices == leaf]
            self.quantile_values[leaf] = np.percentile(leaf_y, self.quantile * 100)

    def predict(self, X):
        leaf_indices = self.tree.apply(X)
        predictions = np.array([self.quantile_values[leaf] for leaf in leaf_indices])
        return predictions

# Optuna Objective Function for Decision Tree
def objective_tree(trial, train_X, train_y, val_X, val_y):
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 50)

    model = QuantileDecisionTreeRegressor(quantile=QUANTILE, min_samples_leaf=min_samples_leaf)
    model.fit(train_X, train_y)
    y_pred = model.predict(val_X)

    return pinball_loss(val_y, y_pred, tau=QUANTILE)

# Complexity parameters
binary_operators = ["+", "*", "/", "-"]
unary_operators = ["exp", "sin", "cos", "log", "square"]
complexity_of_operators = {
    "+": 1,
    "-": 1,
    "*": 1,
    "/": 2,
    "exp": 4,
    "sin": 3,
    "cos": 3,
    "log": 3,
    "square": 2,
}

# results90 storage
results90 = {
    "SQR": {"losses": [], "coverage": [], "complexity": []},
    "LightGBM": {"losses": [], "coverage": []},
    "DecisionTree": {"losses": [], "coverage": [], "complexity": []},
    "LinearQuantile": {"losses": [], "coverage": [], "complexity": []},
}

def process_fold_scores(model_name, fold_scores):
    for metric, scores in fold_scores.items():
        results90[model_name][metric].extend(scores)


# Iterate over datasets
for regression_dataset in regression_dataset_namestry:
    try:
        print(regression_dataset)
        X1, y = fetch_data(regression_dataset, return_X_y=True)
        global_min, global_max = np.min(y), np.max(y)  # Global range for determ. normalization

        X = create_dummy_variables(X1, get_categorical_features(X1))

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

        fold_scores_sqr = {"losses": [], "coverage": [], "complexity": []}
        fold_scores_lgb = {"losses": [], "coverage": []}
        fold_scores_tree = {"losses": [], "coverage": [], "complexity": []}
        fold_scores_linear = {"losses": [], "coverage": [], "complexity": []}

        for train_index, test_index in kf.split(X):
            train_X, test_X = X[train_index], X[test_index]
            train_y, test_y = y[train_index], y[test_index]

            # Symbolic Quantile Regression
            modelq = PySRRegressor(
                niterations=900, #imrpove for better results90
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                complexity_of_operators=complexity_of_operators,
                elementwise_loss="pinball_loss(y_true, y_pred) = max.(0.1 * (y_true - y_pred), (0.1 - 1) * (y_true - y_pred))", #DONT FORGET TO CHANGE WHEN CHANGING QUANTILE (JULIA SYNTAX)
                temp_equation_file=True,
                random_state=SEED
            )

            modelq.fit(train_X, train_y)
            y_pred_symbolic = modelq.predict(test_X)

            # Metrics for SQR
            fold_scores_sqr["losses"].append(normalized_pinball_loss(test_y, y_pred_symbolic, global_min, global_max))
            fold_scores_sqr["coverage"].append(absolute_coverage_error(test_y, y_pred_symbolic))
            fold_scores_sqr["complexity"].append(calculate_expression_complexity(modelq.sympy(), complexity_of_operators))

            # LightGBM Quantile Regression
            study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
            study_lgb.optimize(lambda trial: objective_lgb(trial, train_X, train_y, test_X, test_y), n_trials=10)

            best_params_lgb = study_lgb.best_params
            model_lgb = lgb.LGBMRegressor(objective='quantile', alpha=QUANTILE, **best_params_lgb)
            model_lgb.fit(train_X, train_y)
            y_pred_lgb = model_lgb.predict(test_X)

            # Metrics for LightGBM
            fold_scores_lgb["losses"].append(normalized_pinball_loss(test_y, y_pred_lgb, global_min, global_max))
            fold_scores_lgb["coverage"].append(absolute_coverage_error(test_y, y_pred_lgb))

            # Inside the main loop for dataset processing (NEW)
            study_tree = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
            study_tree.optimize(lambda trial: objective_tree(trial, train_X, train_y, test_X, test_y), n_trials=10)

            best_params_tree = study_tree.best_params  # Get best hyperparameter

            # Train the best Decision Tree model with optimized min_samples_leaf
            model_tree = QuantileDecisionTreeRegressor(quantile=QUANTILE, min_samples_leaf=best_params_tree['min_samples_leaf'])
            model_tree.fit(train_X, train_y)
            y_pred_tree = model_tree.predict(test_X)

            # Metrics for Decision Tree (NEW complexity calculation)
            fold_scores_tree["losses"].append(normalized_pinball_loss(test_y, y_pred_tree, global_min, global_max))
            fold_scores_tree["coverage"].append(absolute_coverage_error(test_y, y_pred_tree))
            fold_scores_tree["complexity"].append(model_tree.tree.tree_.node_count)  # NEW: Store tree complexity

            # Linear Quantile Regression
            study_linear = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
            study_linear.optimize(lambda trial: objective_linear(trial, train_X, train_y, test_X, test_y), n_trials=10)

            best_params_linear = study_linear.best_params
            model_linear = QuantReg(train_y, train_X).fit(q=QUANTILE, max_iter=best_params_linear['max_iter'])
            y_pred_linear = model_linear.predict(test_X)

            # Metrics for Linear Quantile Regression
            fold_scores_linear["losses"].append(normalized_pinball_loss(test_y, y_pred_linear, global_min, global_max))
            fold_scores_linear["coverage"].append(absolute_coverage_error(test_y, y_pred_linear))
            fold_scores_linear["complexity"].append(X.shape[1])

        process_fold_scores("SQR", fold_scores_sqr)
        process_fold_scores("LightGBM", fold_scores_lgb)
        process_fold_scores("DecisionTree", fold_scores_tree)
        process_fold_scores("LinearQuantile", fold_scores_linear)
    except Exception as e:
        print(f"Error processing {regression_dataset}: {e}")

# Display results90
print(results90)


# In[ ]:


# Extract all available metrics dynamically
available_metrics = set()
for model in results90:
    available_metrics.update(results90[model].keys())
metrics = list(available_metrics)  # Ensures only existing metrics are used

models = list(results90.keys())  # ["SQR", "LightGBM", "DecisionTree", "LinearQuantile"]

# Significance level for Bonferroni correction
alpha = 0.05

# Store results
friedman_results = {}
wilcoxon_results = {}

# Perform Friedman test for each available metric
for metric in metrics:
    # Filter models that contain this metric
    valid_models = [model for model in models if metric in results90[model]]

    # Gather data for each valid model
    data = [results90[model][metric] for model in valid_models]

    # Perform Friedman test (only if multiple models have the metric)
    if len(data) > 1:
        stat, p_value = friedmanchisquare(*data)
        friedman_results[metric] = {'statistic': stat, 'p_value': p_value}

        # Check for significance
        wilcoxon_results[metric] = {}  # Initialize results for this metric

        # Perform pairwise Wilcoxon tests with Bonferroni correction
        num_comparisons = len(valid_models) * (len(valid_models) - 1) // 2
        corrected_alpha = alpha / num_comparisons if num_comparisons > 0 else alpha

        for i, model_1 in enumerate(valid_models):
            for j, model_2 in enumerate(valid_models):
                if i < j:  # Avoid duplicate comparisons
                    stat, p_value = wilcoxon(results90[model_1][metric], results90[model_2][metric])
                    wilcoxon_results[metric][f"{model_1} vs {model_2}"] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < corrected_alpha
                    }

# Display results
print("Friedman Test Results:")
for metric, result in friedman_results.items():
    significance = "Significant" if result['p_value'] < alpha else "Not Significant"
    print(f"Metric: {metric}, Statistic: {result['statistic']}, p-value: {result['p_value']} ({significance})")

print("\nWilcoxon Test Results (All Comparisons):")
for metric, comparisons in wilcoxon_results.items():
    print(f"\nMetric: {metric}")
    for comparison, result in comparisons.items():
        significance = "Significant" if result['significant'] else "Not Significant"
        print(f"  {comparison}: Statistic: {result['statistic']}, p-value: {result['p_value']} ({significance})")


# SQR BENCHMARK **50TH** QUANTILE

# In[ ]:


# Set seed for reproducibility
SEED = 42  # Change as needed

# Set NumPy seed
np.random.seed(SEED)

# Set Python random seed
random.seed(SEED)

# Set PyTorch seed (if using)
torch.manual_seed(SEED)

# Set Optuna seed
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce logging clutter
optuna_seed = SEED

# Global quantile setting (EXCEPT FOR PYSR, THIS NEEDS MANUAL ADJUSTMENT)
QUANTILE = 0.5 #(CHANGE PYSR QUANTILE MANUALLY)

# Function to calculate pinball loss
def pinball_loss(y_true, y_pred, tau=QUANTILE):
    residuals = y_true - y_pred
    loss = np.where(residuals >= 0, tau * residuals, (1 - tau) * -residuals)
    return np.mean(loss)

# Function to calculate normalized pinball loss using the global dataset range
def normalized_pinball_loss(y_true, y_pred, global_min, global_max, tau=QUANTILE):
    range_y = global_max - global_min
    loss = pinball_loss(y_true, y_pred, tau)
    return loss / range_y if range_y != 0 else 0  # Avoid division by zero

# Function to calculate absolute coverage error
def absolute_coverage_error(y_true, y_pred, tau=QUANTILE):
    coverage = np.mean(y_pred >= y_true)
    return np.abs(coverage - tau)

# Function to calculate expression complexity
def calculate_expression_complexity(expression, complexity_of_operators):
    try:
        expr = sp.sympify(expression)
    except sp.SympifyError:
        raise ValueError("Invalid expression")

    complexity = 0
    for atom in sp.preorder_traversal(expr):
        if isinstance(atom, sp.Symbol):  # Variables (e.g., x1, x2)
            complexity += 1
        elif isinstance(atom, (int, float, sp.Integer, sp.Float)):  # Constants
            complexity += 1
        elif atom in complexity_of_operators:  # Operators
            complexity += complexity_of_operators[atom]
    return complexity

# LightGBM objective function for Optuna
def objective_lgb(trial, train_X, train_y, val_X, val_y):
    params = {
        'objective': 'quantile',
        'alpha': QUANTILE,
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': SEED,
        'bagging_seed': SEED,
        'feature_fraction_seed': SEED,
        'data_random_seed': SEED,
    }

    if params['min_child_samples'] >= params['num_leaves']:
        raise optuna.exceptions.TrialPruned()

    model = lgb.LGBMRegressor(**params)
    model.fit(train_X, train_y)
    y_pred = model.predict(val_X)
    return pinball_loss(val_y, y_pred, tau=QUANTILE)

# Quantile regression objective function for Optuna
def objective_linear(trial, train_X, train_y, val_X, val_y, tau=QUANTILE):
    max_iter = trial.suggest_int('max_iter', 1000, 5000)
    model = QuantReg(train_y, train_X)
    results50 = model.fit(q=tau, max_iter=max_iter)
    y_pred = results50.predict(val_X)
    return pinball_loss(val_y, y_pred, tau)

# Quantile Decision Tree Regressor
class QuantileDecisionTreeRegressor:
    def __init__(self, quantile=QUANTILE, min_samples_leaf=5, random_state=SEED):
        self.quantile = quantile
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=random_state)

    def fit(self, X, y):
        self.tree.fit(X, y)
        self._add_quantile_info(X, y)

    def _add_quantile_info(self, X, y):
        leaf_indices = self.tree.apply(X)
        unique_leaves = np.unique(leaf_indices)
        self.quantile_values = {}
        for leaf in unique_leaves:
            leaf_y = y[leaf_indices == leaf]
            self.quantile_values[leaf] = np.percentile(leaf_y, self.quantile * 100)

    def predict(self, X):
        leaf_indices = self.tree.apply(X)
        predictions = np.array([self.quantile_values[leaf] for leaf in leaf_indices])
        return predictions

# Optuna Objective Function for Decision Tree
def objective_tree(trial, train_X, train_y, val_X, val_y):
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 50)

    model = QuantileDecisionTreeRegressor(quantile=QUANTILE, min_samples_leaf=min_samples_leaf)
    model.fit(train_X, train_y)
    y_pred = model.predict(val_X)

    return pinball_loss(val_y, y_pred, tau=QUANTILE)

# Complexity parameters
binary_operators = ["+", "*", "/", "-"]
unary_operators = ["exp", "sin", "cos", "log", "square"]
complexity_of_operators = {
    "+": 1,
    "-": 1,
    "*": 1,
    "/": 2,
    "exp": 4,
    "sin": 3,
    "cos": 3,
    "log": 3,
    "square": 2,
}

# results50 storage
results50 = {
    "SQR": {"losses": [], "coverage": [], "complexity": []},
    "LightGBM": {"losses": [], "coverage": []},
    "DecisionTree": {"losses": [], "coverage": [], "complexity": []},
    "LinearQuantile": {"losses": [], "coverage": [], "complexity": []},
}

def process_fold_scores(model_name, fold_scores):
    for metric, scores in fold_scores.items():
        results50[model_name][metric].extend(scores)


# Iterate over datasets
for regression_dataset in regression_dataset_namestry:
    try:
        print(regression_dataset)
        X1, y = fetch_data(regression_dataset, return_X_y=True)
        global_min, global_max = np.min(y), np.max(y)  # Global range for determ. normalization

        X = create_dummy_variables(X1, get_categorical_features(X1))

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

        fold_scores_sqr = {"losses": [], "coverage": [], "complexity": []}
        fold_scores_lgb = {"losses": [], "coverage": []}
        fold_scores_tree = {"losses": [], "coverage": [], "complexity": []}
        fold_scores_linear = {"losses": [], "coverage": [], "complexity": []}

        for train_index, test_index in kf.split(X):
            train_X, test_X = X[train_index], X[test_index]
            train_y, test_y = y[train_index], y[test_index]

            # Symbolic Quantile Regression
            modelq = PySRRegressor(
                niterations=900, #imrpove for better results50
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                complexity_of_operators=complexity_of_operators,
                elementwise_loss="pinball_loss(y_true, y_pred) = max.(0.5 * (y_true - y_pred), (0.5 - 1) * (y_true - y_pred))", #DONT FORGET TO CHANGE WHEN CHANGING QUANTILE (JULIA SYNTAX)
                temp_equation_file=True,
                random_state=SEED
            )

            modelq.fit(train_X, train_y)
            y_pred_symbolic = modelq.predict(test_X)

            # Metrics for SQR
            fold_scores_sqr["losses"].append(normalized_pinball_loss(test_y, y_pred_symbolic, global_min, global_max))
            fold_scores_sqr["coverage"].append(absolute_coverage_error(test_y, y_pred_symbolic))
            fold_scores_sqr["complexity"].append(calculate_expression_complexity(modelq.sympy(), complexity_of_operators))

            # LightGBM Quantile Regression
            study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
            study_lgb.optimize(lambda trial: objective_lgb(trial, train_X, train_y, test_X, test_y), n_trials=10)

            best_params_lgb = study_lgb.best_params
            model_lgb = lgb.LGBMRegressor(objective='quantile', alpha=QUANTILE, **best_params_lgb)
            model_lgb.fit(train_X, train_y)
            y_pred_lgb = model_lgb.predict(test_X)

            # Metrics for LightGBM
            fold_scores_lgb["losses"].append(normalized_pinball_loss(test_y, y_pred_lgb, global_min, global_max))
            fold_scores_lgb["coverage"].append(absolute_coverage_error(test_y, y_pred_lgb))

            # Inside the main loop for dataset processing (NEW)
            study_tree = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
            study_tree.optimize(lambda trial: objective_tree(trial, train_X, train_y, test_X, test_y), n_trials=10)

            best_params_tree = study_tree.best_params  # Get best hyperparameter

            # Train the best Decision Tree model with optimized min_samples_leaf
            model_tree = QuantileDecisionTreeRegressor(quantile=QUANTILE, min_samples_leaf=best_params_tree['min_samples_leaf'])
            model_tree.fit(train_X, train_y)
            y_pred_tree = model_tree.predict(test_X)

            # Metrics for Decision Tree (NEW complexity calculation)
            fold_scores_tree["losses"].append(normalized_pinball_loss(test_y, y_pred_tree, global_min, global_max))
            fold_scores_tree["coverage"].append(absolute_coverage_error(test_y, y_pred_tree))
            fold_scores_tree["complexity"].append(model_tree.tree.tree_.node_count)  # NEW: Store tree complexity

            # Linear Quantile Regression
            study_linear = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
            study_linear.optimize(lambda trial: objective_linear(trial, train_X, train_y, test_X, test_y), n_trials=10)

            best_params_linear = study_linear.best_params
            model_linear = QuantReg(train_y, train_X).fit(q=QUANTILE, max_iter=best_params_linear['max_iter'])
            y_pred_linear = model_linear.predict(test_X)

            # Metrics for Linear Quantile Regression
            fold_scores_linear["losses"].append(normalized_pinball_loss(test_y, y_pred_linear, global_min, global_max))
            fold_scores_linear["coverage"].append(absolute_coverage_error(test_y, y_pred_linear))
            fold_scores_linear["complexity"].append(X.shape[1])

        process_fold_scores("SQR", fold_scores_sqr)
        process_fold_scores("LightGBM", fold_scores_lgb)
        process_fold_scores("DecisionTree", fold_scores_tree)
        process_fold_scores("LinearQuantile", fold_scores_linear)
    except Exception as e:
        print(f"Error processing {regression_dataset}: {e}")

# Display results50
print(results50)


# In[ ]:


# Extract all available metrics dynamically
available_metrics = set()
for model in results50:
    available_metrics.update(results50[model].keys())
metrics = list(available_metrics)  # Ensures only existing metrics are used

models = list(results50.keys())  # ["SQR", "LightGBM", "DecisionTree", "LinearQuantile"]

# Significance level for Bonferroni correction
alpha = 0.05

# Store results
friedman_results = {}
wilcoxon_results = {}

# Perform Friedman test for each available metric
for metric in metrics:
    # Filter models that contain this metric
    valid_models = [model for model in models if metric in results50[model]]

    # Gather data for each valid model
    data = [results50[model][metric] for model in valid_models]

    # Perform Friedman test (only if multiple models have the metric)
    if len(data) > 1:
        stat, p_value = friedmanchisquare(*data)
        friedman_results[metric] = {'statistic': stat, 'p_value': p_value}

        # Check for significance
        wilcoxon_results[metric] = {}  # Initialize results for this metric

        # Perform pairwise Wilcoxon tests with Bonferroni correction
        num_comparisons = len(valid_models) * (len(valid_models) - 1) // 2
        corrected_alpha = alpha / num_comparisons if num_comparisons > 0 else alpha

        for i, model_1 in enumerate(valid_models):
            for j, model_2 in enumerate(valid_models):
                if i < j:  # Avoid duplicate comparisons
                    stat, p_value = wilcoxon(results50[model_1][metric], results50[model_2][metric])
                    wilcoxon_results[metric][f"{model_1} vs {model_2}"] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < corrected_alpha
                    }

# Display results
print("Friedman Test Results:")
for metric, result in friedman_results.items():
    significance = "Significant" if result['p_value'] < alpha else "Not Significant"
    print(f"Metric: {metric}, Statistic: {result['statistic']}, p-value: {result['p_value']} ({significance})")

print("\nWilcoxon Test Results (All Comparisons):")
for metric, comparisons in wilcoxon_results.items():
    print(f"\nMetric: {metric}")
    for comparison, result in comparisons.items():
        significance = "Significant" if result['significant'] else "Not Significant"
        print(f"  {comparison}: Statistic: {result['statistic']}, p-value: {result['p_value']} ({significance})")


# In[17]:


# Merge both results into a dictionary with dataset labels
all_results = {"results90": results90, "results50": results50}

# Create a structured dictionary for easy merging
data_list = []
num_entries = len(next(iter(results90.values()))["losses"])  # Assume same number of entries for all models

for dataset, results in all_results.items():
    for i in range(num_entries):  # Iterate over the index of each loss/coverage/complexity entry
        row = {"Dataset": dataset}
        for model, metrics in results.items():
            row[f"{model}Loss"] = float(metrics["losses"][i])
            row[f"{model}Coverage"] = float(metrics["coverage"][i])
            row[f"{model}Complexity"] = metrics.get("complexity", [None] * num_entries)[i]
        data_list.append(row)

# Convert list to DataFrame
df = pd.DataFrame(data_list)

# Save to CSV with correct decimal formatting
df.to_csv("results.csv", index=False, sep=",", decimal=".")

print("CSV file saved successfully.")


# In[18]:


with open("results.csv", "r", encoding="utf-8") as f:
    for _ in range(10):  # Print first 5 lines
        print(f.readline())

