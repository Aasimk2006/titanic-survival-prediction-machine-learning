# Titanic Survival Prediction — Project 3

A classification project that predicts Titanic passenger survival using feature engineering, missing-value handling, categorical encoding, model comparison, and feature-importance explainability.

## Dataset

This project uses the exact `titanic.csv` supplied for the assigned Kaggle dataset.

The supplied file contains 418 rows and 12 columns, including `Survived` as the target.

## Requirements covered

- Extract passenger **Title** from `Name`
- Create **FamilySize**
- Create **IsAlone**
- Create **CabinKnown** to represent cabin presence/missingness
- Handle missing **Age** with median imputation
- Handle missing **Fare** with median imputation
- Handle missing categorical values with most-frequent imputation
- One-hot encode categorical variables
- Compare Logistic Regression, Random Forest, and Gradient Boosting
- Report Accuracy, Precision, Recall, F1-score, and confusion matrix
- Explain the Random Forest using feature importance
- Save the complete trained pipeline with joblib
- Provide a standalone inference example

## Important dataset observation

The supplied 418-row file has a deterministic relationship between `Sex` and `Survived`: every female row is labeled `1` and every male row is labeled `0`.

Therefore, the 100% holdout scores produced by the tested models are a property of this supplied dataset. They should **not** be presented as 100% general-world Titanic prediction performance.

## Project structure

```text
titanic_survival_prediction_project/
├── data/
│   └── titanic.csv
├── models/
│   └── titanic_best_model.joblib
├── titanic_survival_prediction.ipynb
├── preprocessing.py
├── inference.py
├── README.md
└── requirements.txt
```

## Run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Open `titanic_survival_prediction.ipynb` in VS Code/Jupyter and run all cells.

The notebook creates:

```text
models/titanic_best_model.joblib
```

Run the standalone inference example from the project root:

```bash
python inference.py
```

The saved model contains the feature-engineering and preprocessing pipeline together with the classifier, so raw passenger columns can be supplied directly during inference.

## Explainability

The project uses Random Forest `feature_importances_` for model explanation. Larger values indicate features that contributed more to the forest's impurity reduction; feature importance is an explanation aid, not a causal claim.
