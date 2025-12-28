import pandas as pd
import numpy as np

class CustomerChurnPreprocessor:
    def __init__(self):
        self.drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'Complain']
        self.cat_cols = ['Geography', 'Gender', 'Card Type']
        
    def preprocess(self, df):
        df_processed = df.copy()
        
        cols_to_drop = [c for c in self.drop_cols if c in df_processed.columns]
        df_processed = df_processed.drop(columns=cols_to_drop)
        
        df_processed = self._features_engineering(df_processed)
        df_processed = self._encode_categoricals(df_processed)
        
        return df_processed

    def _features_engineering(self, df):
        df['BalanceSalaryRatio'] = (df['Balance'] / df['EstimatedSalary']) + 1e-9
        
        df['TenureByAge'] = df['Tenure'] / (df['Age'] + 1e-9)
        
        df['HasBalance'] = df['Balance'].apply(lambda x: 1 if x > 0 else 0)
        
        df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + 1e-9)
        
        return df

    def _encode_categoricals(self, df):
        df = pd.get_dummies(df, columns=self.cat_cols, drop_first=True)
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)
        
        return df