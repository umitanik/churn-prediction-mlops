import pandas as pd

class DataAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def show_all(self):
        print(f"\n{'#'*20} GENERAL OVERVIEW {'#'*20}")
        self.get_general_info()

        print(f"\n{'#'*20} DUPLICATES {'#'*20}")
        self.check_duplicates()

        print(f"\n{'#'*20} FEATURE TYPES {'#'*20}")
        num_cols, cat_cols = self.check_feature_types()

        print(f"\n{'#'*20} MISSING VALUES {'#'*20}")
        self.check_missing_values()

        print(f"\n{'#'*20} COLUMN SUMMARY {'#'*20}")
        self.check_columns_summary()

        print(f"\n{'#'*20} STATISTICS {'#'*20}")
        
        if num_cols:
            print("\n--- Numerical Summary ---")
            print(self.summarize_numerical())
        
        if cat_cols:
            print("\n--- Categorical Summary ---")
            print(self.summarize_categorical())

    def get_general_info(self):
        print(f"Veri Seti Boyutu: {self.data.shape}")
        
    def check_columns_summary(self):
        summary_df = pd.DataFrame({
            'Dtype': self.data.dtypes,
            'N_Unique': self.data.nunique(),
            'Missing_Rate (%)': self.data.isnull().mean() * 100
        })
        
        print(summary_df.sort_values(by='Dtype'))
        return summary_df

    def check_missing_values(self):
        missing_count = self.data.isnull().sum()
        missing_pct = 100 * self.data.isnull().mean()
        missing_df = pd.concat([missing_count, missing_pct], axis=1, keys=['Missing Count', 'Percentage'])
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage', ascending=False)
        
        if missing_df.empty:
            print("Veri setinde hiç eksik değer yok.")
        else:
            print(missing_df)
            
        return missing_df

    def check_feature_types(self):
        num_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numerical Features ({len(num_cols)}): {num_cols}")
        print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
        print("="*50)
        
        return num_cols, cat_cols

    def check_duplicates(self):
        num_duplicates = self.data.duplicated().sum()
        print(f"Tekrar eden satır sayısı: {num_duplicates}")
        return num_duplicates

    def summarize_numerical(self):
        return self.data.describe().T 

    def summarize_categorical(self):
        return self.data.describe(include=['object']).T