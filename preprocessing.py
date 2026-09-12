from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create Titanic-specific features required by the project."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "Name" in X.columns:
            title = (
                X["Name"].fillna("").astype(str)
                .str.extract(r",\s*([^.]*)\.", expand=False)
                .str.strip().replace("", "Unknown")
            )
            common_titles = {"Mr", "Miss", "Mrs", "Master"}
            X["Title"] = title.where(title.isin(common_titles), "Rare")

        if "SibSp" in X.columns and "Parch" in X.columns:
            X["FamilySize"] = X["SibSp"].fillna(0) + X["Parch"].fillna(0) + 1
            X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        if "Cabin" in X.columns:
            X["CabinKnown"] = X["Cabin"].notna().astype(int)

        drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
        return X
