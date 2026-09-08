from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CAT_COLS = ['Geography', 'Gender', 'Card Type']


def build_encoder():
    """One-hot encoder for the categorical columns, as a fitted pipeline step.

    Encoding belongs inside the model pipeline rather than in preprocess().
    The list of categories is a property of the training data, so it has to be
    learned during fit and travel with the saved artifact. A stateless
    pd.get_dummies derives its output columns from whatever frame it is handed,
    which silently produces different columns for a single-row inference
    request than for the training set.
    """
    return ColumnTransformer(
        transformers=[
            (
                'cat',
                OneHotEncoder(
                    drop='first',
                    handle_unknown='error',
                    sparse_output=False,
                ),
                CAT_COLS,
            )
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )


class CustomerChurnPreprocessor:
    """Row-wise preparation only: dropping columns and deriving features.

    Every step here depends on a single row, so one row and a full batch are
    treated identically. Anything that has to be learned from the training set
    lives in the model pipeline instead - see build_encoder().
    """

    def __init__(self):
        self.drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'Complain']
        self.cat_cols = CAT_COLS

    def preprocess(self, df):
        df_processed = df.copy()

        cols_to_drop = [c for c in self.drop_cols if c in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop)

        df_processed = self._features_engineering(df_processed)

        return df_processed

    def _features_engineering(self, df):
        df['BalanceSalaryRatio'] = (df['Balance'] / df['EstimatedSalary']) + 1e-9

        df['TenureByAge'] = df['Tenure'] / (df['Age'] + 1e-9)

        df['HasBalance'] = df['Balance'].apply(lambda x: 1 if x > 0 else 0)

        df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + 1e-9)

        return df
