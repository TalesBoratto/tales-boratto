---
title: River Phosphate Prediction with Optimized Machine Learning
summary: Comparing four regression models and predictor configurations for phosphate concentration estimation using Harris Hawks Optimization and nested cross-validation.
tags:
  - Regression
  - Environmental Data
  - Water Quality
  - Hyperparameter Optimization
  - Leave-One-Out
  - Nested Cross-Validation
  - Feature Importance
  - Scikit-learn
  - Machine Learning
featured: true
draft: false
---

## Overview

This project investigates phosphate concentration prediction in river systems using environmental measurements and supervised machine learning.

The study compares four regression models combined with Harris Hawks Optimization (HHO): Elastic Net, Extreme Learning Machine, Gradient Boosting, and Support Vector Regression.

The workflow evaluates four predictor configurations using an outer leave-one-out procedure and five-fold cross-validation for hyperparameter selection within each training partition.

Beyond predictive performance, the analysis examines selected hyperparameters and the contributions of individual variables to the models' predictions.

## Associated Publication

This project is associated with the following peer-reviewed article:

Boratto, T. H. A., Campos, D. E. D., Fonseca, D. L., Soares Filho, W. A., Yaseen, Z. M., Gorgoglione, A., and Goliatt, L. (2024).

Hybridized machine learning models for phosphate pollution modeling in water systems for multiple uses. Journal of Water Process Engineering, 64, 105598.

[Read the article](https://doi.org/10.1016/j.jwpe.2024.105598)

## Data Availability

The dataset is not distributed with this portfolio.

This page presents the methodology and selected results. Reproducing the experiments requires access to the corresponding data and experimental settings.

## Problem Context

The study addresses phosphate concentration estimation
in freshwater systems connected to the headwaters of
the Cantareira water supply system in Brazil.

Rather than predicting a future concentration at a specified
forecast horizon, the modeling task estimates phosphate
concentration from environmental measurements and
watershed information.

The central questions are:

- How do different regression models perform on this dataset?
- How does the predictor configuration affect model performance?
- Which variables contribute most strongly to the predictions?

## Dataset

### Study Area and Observations

The study uses freshwater quality data extracted and adapted from Taffarello et al.

The observations were collected mainly in three Brazilian municipalities:

- Extrema, Minas Gerais.
- Joanópolis, São Paulo.
- Nazaré Paulista, São Paulo.

The experimental dataset contains 62 observations, seven candidate predictors, and one target variable.

### Variables

| ID | Variable | Unit |
|---|---|---|
| X1 | Flow rate | m³/s |
| X2 | pH | — |
| X3 | Turbidity | NTU |
| X4 | Electrical conductivity | µS/cm |
| X5 | Dissolved solids | mg/L |
| X6 | Watershed area | km² |
| X7 | Apparent color | Not reported |
| y | Phosphate concentration | mg/L |

The observed phosphate concentrations range from 0.020 to 0.390 mg/L, with a mean of 0.132 mg/L.

### Predictor Configurations

Four input configurations are evaluated using subsets of the same dataset.

| Configuration | Number of predictors | Variables |
|---|---:|---|
| M1 | 7 | X1, X2, X3, X4, X5, X6, X7 |
| M2 | 5 | X1, X3, X5, X6, X7 |
| M3 | 4 | X3, X5, X6, X7 |
| M4 | 3 | X3, X5, X7 |

The configurations were defined through a manual, correlation-based screening procedure.

M1 includes all candidate predictors. M2 removes pH and electrical conductivity; M3 additionally removes flow rate; and M4 also removes watershed area.

This comparison examines how changes in the available predictors affect each model.

## Methodology

### 1. Dataset Preparation

- Assemble the environmental predictors and phosphate measurements.
- Define the observations included in the study.
- Construct the four predictor configurations.
- Apply the same modeling framework to each configuration.

### 2. Outer Leave-One-Out Evaluation

The evaluation uses leave-one-out to generate the outer
training and test partitions.

For each of the 62 outer partitions:

- One observation is reserved for testing.
- The remaining 61 observations form the training set.
- Hyperparameter selection is performed within that training set.
- The selected model is fitted on the outer training set.
- A prediction is generated for the held-out observation.

### 3. Inner Hyperparameter Selection

Within each outer training partition, HHO searches for model-specific hyperparameter configurations.

Candidate configurations are evaluated using five-fold cross-validation on the current training data.

The outer test observation remains separate from hyperparameter selection.

### 4. Comparative Analysis

The results are examined across:

- Four regression model families.
- Four predictor configurations.
- Selected hyperparameter distributions.
- Feature contributions estimated using Shapley values.

## Regression Models

### Elastic Net

Elastic Net provides a regularized linear regression approach
combining L1 and L2 penalties.

The search includes the regularization coefficient, the L1 ratio, and whether an intercept is fitted.

### Extreme Learning Machine

The Extreme Learning Machine uses a randomly initialized hidden layer to transform the input variables.

The output layer is fitted using Ridge regression when the selected L2 penalty is positive. When the penalty is zero, output weights are obtained using the pseudoinverse.

The hyperparameter search considers the number of hidden neurons, the activation function, and the L2 penalty.

### Gradient Boosting

Gradient Boosting builds an additive ensemble of trees.

The search includes the learning rate, number of estimators, subsampling fraction, and maximum tree depth.

### Support Vector Regression

Support Vector Regression uses kernel-based regression to represent linear or nonlinear relationships.

The search includes the regularization parameter, kernel coefficient, polynomial degree, and kernel type.

## Harris Hawks Optimization

HHO provides a common mechanism for searching the hyperparameter spaces of the four model families.

Each candidate solution represents a possible configuration of the corresponding regression model.

The optimizer explores candidate configurations and refines promising regions of the search space using performance evaluated within the training data.

## Performance Evaluation

The study reports four main predictive performance metrics:

- Pearson correlation coefficient, R.
- Coefficient of determination, R².
- Root mean squared error, RMSE.
- Mean absolute error, MAE.

RMSE and MAE are reported in mg/L.

Results are compared within each predictor configuration rather than treating the configurations as independent datasets.


## Published Results

The results summarized below are reported in the associated peer-reviewed article. Values in parentheses represent the reported standard deviations.

### Full Predictor Configuration: M1

| Model | R | R² | RMSE (mg/L) | MAE (mg/L) |
|---|---:|---:|---:|---:|
| HHO-ELM | 0.815 (0.018) | 0.652 (0.042) | 0.050 (0.003) | 0.038 (0.002) |
| HHO-EN | 0.824 (0.008) | 0.670 (0.015) | 0.049 (0.001) | 0.037 (0.001) |
| HHO-GB | 0.791 (0.025) | 0.619 (0.044) | 0.053 (0.003) | 0.041 (0.003) |
| HHO-SVR | 0.802 (0.005) | 0.643 (0.009) | 0.051 (0.001) | 0.038 (0.000) |

HHO-EN achieves the best reported mean values across all four metrics for the full predictor configuration.

### Effect of Predictor Configuration

The effect of removing predictors depends on the model.

For HHO-EN, mean RMSE remains between 0.049 and 0.050 mg/L across the four configurations.

For HHO-GB, mean R² increases from 0.619 with M1 to 0.664 with M3. This illustrates that using fewer predictors can improve performance for a particular model.

For HHO-SVR, moving from M3 to M4 reduces mean R² from 0.634 to 0.501 and increases mean RMSE from 0.052 to 0.060 mg/L.

These results show that predictor reduction does not affect every model in the same way.


## Hyperparameter Analysis

The study summarizes the hyperparameters selected by HHO for each model and predictor configuration.

The table reports approximate median numerical values and representative categorical settings.

| Model | Hyperparameter | M1 | M2 | M3 | M4 |
|---|---|---:|---:|---:|---:|
| HHO-ELM | L2 penalty (C₂) | 250 | 230 | 280 | 430 |
| HHO-ELM | Hidden neurons | 9 | 8 | 5 | 7 |
| HHO-ELM | Activation | Identity | Identity | Identity | Identity |
| HHO-EN | L1 ratio | 0 | 0 | 0.025 | 0.025 |
| HHO-EN | α | 0.7 | 0.8 | 7 | 6 |
| HHO-EN | Fit intercept | False | True | True | True |
| HHO-GB | Learning rate | 0.11 | 0.11 | 0.095 | 0.1000 |
| HHO-GB | Maximum depth | 6 | 6 | 5 | 5 |
| HHO-GB | Number of estimators | 143 | 145 | 135 | 122 |
| HHO-GB | Subsample | 0.50 | 0.50 | 0.40 | 0.45 |
| HHO-SVR | C | 26137 | 28152 | 27578 | 0.001 |
| HHO-SVR | γ | 2.0 | 2.5 | 2.5 | 0.002 |
| HHO-SVR | Degree | 1 | 1 | 1 | 1 |
| HHO-SVR | Kernel | Polynomial | Polynomial | Polynomial | Linear |

The ELM configurations are characterized by small hidden layers and identity activation. For Elastic Net, the reported L1 ratios are zero or close to zero.

Gradient Boosting exhibits relatively similar representative settings across configurations, with maximum depths of five or six and between 122 and 145 estimators.

SVR shows a more pronounced change in M4, where the representative kernel changes from polynomial to linear and the reported regularization parameter C decreases substantially.

These summaries describe the selected parameter distributions. They do not represent a single parameter vector used for every fitted model.

## Feature Importance

The Shapley-based analysis compares feature contributions
across the four regression models and predictor configurations.

![Average Shapley values across regression models and predictor configurations](shap-bar.png)

Apparent color (X7) is the most influential predictor
in the reported analysis. Watershed area (X6) is also
prominent in the configurations where it is included,
while dissolved solids (X5) contributes across configurations.

The figure also illustrates model-dependent differences:
the relative contributions of the remaining predictors
vary across regressors and input configurations.

These findings complement the performance comparison
by showing which inputs influence model predictions.

Feature contributions describe model behavior within
the studied data and should not be interpreted as
evidence of causal effects.

## Implementation

The following code excerpts illustrate predictor configuration, model construction, hyperparameter optimization, and nested evaluation.

The workflow uses leave-one-out for the outer evaluation and five-fold cross-validation for hyperparameter selection within each training partition.

The excerpts assume that `data` is a pandas DataFrame containing the seven numerical predictors and the phosphate concentration target. Dataset loading is omitted because the data are not distributed with this portfolio.

```python
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    cross_val_score,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mealpy import FloatVar
from mealpy.swarm_based import HHO

from util.ELM import ELMRegressor
```

The ELM model uses the project's random-layer implementation. Its supporting classes and dataset loading are omitted from the excerpts for brevity.

### Predictor Configuration Definition

Four configurations are defined using subsets of the same seven candidate predictors.

A shared set of complete observations is used across the configurations so that the comparisons retain the same target values and outer evaluation partitions.

```python
FEATURE_SETS = {
    "M1": ["X1", "X2", "X3", "X4", "X5", "X6", "X7"],
    "M2": ["X1", "X3", "X5", "X6", "X7"],
    "M3": ["X3", "X5", "X6", "X7"],
    "M4": ["X3", "X5", "X7"],
}

FEATURE_NAMES = {
    "X1": "Flow rate",
    "X2": "pH",
    "X3": "Turbidity",
    "X4": "Electrical conductivity",
    "X5": "Dissolved solids",
    "X6": "Watershed area",
    "X7": "Apparent color",
}


def prepare_dataset(data):
    columns = FEATURE_SETS["M1"] + ["y"]

    clean = (
        data.loc[:, columns]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    X = clean.loc[:, FEATURE_SETS["M1"]]
    y = clean["y"].to_numpy(dtype=float)

    return X, y
```

### Model Construction and Search Spaces

Each model has its own hyperparameter representation. Integer and categorical values are decoded from the candidate vectors proposed by the optimizer.

The search spaces below define the hyperparameters considered
for each model family.

```python
ACTIVATIONS = [
    "identity",
    "gaussian",
    "multiquadric",
    "inv_multiquadric",
    "relu",
    "swish",
    "sigmoid",
]

KERNELS = ["linear", "poly", "rbf", "sigmoid"]

SEARCH_SPACES = {
    # hidden neurons, activation, L2 penalty
    "ELM": [
        (1, 300),
        (0, 6),
        (0.0, 1000.0),
    ],

    # alpha, L1 ratio, fit intercept
    "EN": [
        (0.0, 100.0),
        (0.0, 1.0),
        (0, 1),
    ],

    # learning rate, estimators, subsample, maximum depth
    "GB": [
        (1e-5, 1.0),
        (1, 300),
        (0.1, 1.0),
        (1, 12),
    ],

    # C, gamma, degree, kernel
    "SVR": [
        (0.001, 100000.0),
        (0.001, 10.0),
        (1, 3),
        (0, 3),
    ],
}


def decode_parameters(name, solution):
    p = np.asarray(solution, dtype=float)

    if name == "ELM":
        return {
            "n_hidden": int(round(p[0])),
            "activation_func": ACTIVATIONS[int(round(p[1]))],
            "l2": int(p[2] * 1000) / 1000.0,
        }

    if name == "EN":
        return {
            "alpha": float(p[0]),
            "l1_ratio": float(p[1]),
            "fit_intercept": bool(round(p[2])),
        }

    if name == "GB":
        return {
            "learning_rate": float(p[0]),
            "n_estimators": int(round(p[1])),
            "subsample": float(p[2]),
            "max_depth": int(round(p[3])),
        }

    if name == "SVR":
        return {
            "C": float(p[0]),
            "gamma": float(p[1]),
            "degree": int(round(p[2])),
            "kernel": KERNELS[int(round(p[3]))],
        }

    raise ValueError(f"Unknown model: {name}")


def build_model(name, solution, seed):
    parameters = decode_parameters(name, solution)

    if name == "ELM":
        penalty = parameters["l2"]

        output_regressor = (
            None
            if penalty < 1e-4
            else Ridge(
                alpha=penalty,
                random_state=seed,
            )
        )

        return ELMRegressor(
            n_hidden=parameters["n_hidden"],
            activation_func=parameters["activation_func"],
            regressor=output_regressor,
            alpha=1,
            random_state=seed,
        )

    if name == "EN":
        return ElasticNet(
            **parameters,
            max_iter=20000,
            random_state=seed,
        )

    if name == "GB":
        return GradientBoostingRegressor(
            **parameters,
            random_state=seed,
        )

    if name == "SVR":
        return SVR(**parameters)

    raise ValueError(f"Unknown model: {name}")
```

### Inner Hyperparameter Optimization

Within each outer training partition, candidate configurations are evaluated using five-fold cross-validation.

The objective below minimizes the mean validation RMSE. After the search, the selected configuration is fitted on the complete outer training partition.

```python
def optimize_model(
    name,
    X_train,
    y_train,
    inner_splits,
    seed,
    epochs=20,
    population_size=20,
):
    bounds = SEARCH_SPACES[name]

    def objective(solution):
        model = build_model(name, solution, seed)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=inner_splits,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        )

        return -float(np.mean(scores))

    problem = {
        "bounds": FloatVar(
            lb=tuple(lower for lower, _ in bounds),
            ub=tuple(upper for _, upper in bounds),
        ),
        "obj_func": objective,
        "minmax": "min",
        "log_to": None,
    }

    optimizer = HHO.OriginalHHO(
        epoch=epochs,
        pop_size=population_size,
    )

    best = optimizer.solve(problem, seed=seed)

    model = build_model(name, best.solution, seed)
    model.fit(X_train, y_train)

    return model, {
        "parameters": decode_parameters(name, best.solution),
        "inner_cv_rmse": float(best.target.fitness),
    }
```

### Outer Leave-One-Out Evaluation

Each outer iteration holds out one observation for testing.

The remaining observations are used for hyperparameter selection and model fitting. All models and predictor configurations share the same outer partitions.

The implementation stores held-out predictions, selected parameters, and inner cross-validation scores.

```python
def run_nested_evaluation(
    X,
    y,
    seeds=(42,),
    epochs=20,
    population_size=20,
):
    y = np.asarray(y, dtype=float).ravel()

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if len(X) < 6:
        raise ValueError(
            "At least six observations are required for this setup."
        )

    predictions = []
    searches = []

    outer_splits = list(LeaveOneOut().split(X))

    for repeat, seed in enumerate(seeds, start=1):
        for fold, (train_idx, test_idx) in enumerate(
            outer_splits, start=1
        ):
            fold_seed = int(seed)

            inner_cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=fold_seed,
            )

            inner_splits = list(inner_cv.split(train_idx))

            y_train = y[train_idx]
            observed = float(y[test_idx[0]])

            for configuration, columns in FEATURE_SETS.items():
                X_configuration = X.loc[:, columns].to_numpy(
                    dtype=float
                )

                X_train = X_configuration[train_idx]
                X_test = X_configuration[test_idx]

                for name in SEARCH_SPACES:
                    model, search = optimize_model(
                        name=name,
                        X_train=X_train,
                        y_train=y_train,
                        inner_splits=inner_splits,
                        seed=fold_seed,
                        epochs=epochs,
                        population_size=population_size,
                    )

                    predicted = float(model.predict(X_test)[0])

                    identifiers = {
                        "Repeat": repeat,
                        "Seed": int(seed),
                        "Fold": fold,
                        "Configuration": configuration,
                        "Model": name,
                    }

                    predictions.append({
                        **identifiers,
                        "Observation": int(test_idx[0]),
                        "Observed": observed,
                        "Predicted": predicted,
                    })

                    searches.append({
                        **identifiers,
                        **search,
                    })

    return pd.DataFrame(predictions), pd.DataFrame(searches)
```

### Prediction and Metric Aggregation

Metrics are calculated from the complete collection of held-out predictions for each model and predictor configuration.

When multiple complete evaluations are performed, their metrics are summarized separately across repetitions.

```python
def calculate_metrics(group):
    observed = group["Observed"].to_numpy(dtype=float)
    predicted = group["Predicted"].to_numpy(dtype=float)

    if np.std(observed) == 0 or np.std(predicted) == 0:
        correlation = np.nan
    else:
        correlation = float(
            np.corrcoef(observed, predicted)[0, 1]
        )

    return {
        "R": correlation,
        "R2": r2_score(observed, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(observed, predicted)
        ),
        "MAE": mean_absolute_error(observed, predicted),
    }


def summarize_predictions(predictions):
    identifiers = [
        "Repeat",
        "Seed",
        "Configuration",
        "Model",
    ]

    records = []

    for key, group in predictions.groupby(
        identifiers, sort=False
    ):
        records.append({
            **dict(zip(identifiers, key)),
            **calculate_metrics(group),
        })

    metrics = pd.DataFrame(records)

    summary = (
        metrics.groupby(["Configuration", "Model"])[
            ["R", "R2", "RMSE", "MAE"]
        ]
        .agg(["mean", "std"])
    )

    return metrics, summary
```

## Discussion

### Model Complexity and Dataset Size

The performance of Elastic Net shows that a regularized linear model can be competitive with nonlinear alternatives for this particular prediction task.

Model complexity alone does not determine the ranking observed in the experiments.

### Predictor Selection

The four configurations reveal model-dependent responses to predictor removal.

The improvement observed for Gradient Boosting with M3 contrasts with the decline observed for SVR when watershed area is removed in M4.

The comparison therefore supports examining predictor choice together with model choice.

### Practical Interpretation

The workflow investigates the use of environmental measurements to estimate phosphate concentration.

The results support further study of data-driven tools for water-quality assessment, while operational use would require evaluation under the intended deployment conditions.

## Limitations and Reproducibility

### Dataset Scope

The analysis uses 62 observations from the study region. The reported results do not, by themselves, establish equivalent performance in other watersheds or sampling conditions.

### Evaluation Scope

The outer leave-one-out procedure evaluates held-out observations from the available dataset.

It should not be presented as a prospective forecasting experiment or as validation on an independent river system.

### Predictor Configuration Scope

M1–M4 are predefined predictor configurations. The comparison does not represent an exhaustive search over all possible feature subsets.

### Data Availability

The dataset is not distributed with this portfolio. Independent reproduction of the numerical results is therefore not possible from this page alone.

## Conclusion

This project combines environmental regression, metaheuristic hyperparameter optimization, and nested model evaluation to investigate phosphate concentration prediction.

Elastic Net provides strong reported performance across the predictor configurations, while the Gradient Boosting results illustrate the importance of considering model choice and predictor selection
together.

The case study also examines feature contributions and selected hyperparameters, connecting numerical performance with interpretation of model behavior.