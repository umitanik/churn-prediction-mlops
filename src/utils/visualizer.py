import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from src.utils.eda import DataAnalyzer

class DataVisualizer(DataAnalyzer):
    def __init__(self, data: pd.DataFrame, target_col: str = None):
        super().__init__(data)
        self.target_col = target_col
        sns.set_theme(style="whitegrid", palette="Set2")

    def _get_grid_params(self, features, n_cols=2):
        n_features = len(features)
        n_rows = math.ceil(n_features / n_cols)
        figsize = (20, 5 * n_rows)
        return n_rows, n_cols, figsize

    def plot_correlation_heatmap(self):
        num_cols, _ = self.check_feature_types()
        corr = self.data[num_cols].corr().round(2)
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt='.2f', square=True, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap", fontsize=20, fontweight='bold')
        plt.show()

    def plot_categorical_distributions(self, limit=15):
        _, cat_cols = self.check_feature_types()
        
        plot_cols = [c for c in cat_cols if self.data[c].nunique() <= limit]
        
        if not plot_cols:
            print("No suitable categorical variables to plot.")
            return 
        
        n_rows, n_cols, figsize = self._get_grid_params(plot_cols)
        plt.figure(figsize=figsize)
        plt.suptitle("Categorical Variables Distribution", fontsize=20, fontweight='bold')

        for i, col in enumerate(plot_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.title(f'Variable {col}')
            sns.countplot(data=self.data, x=col, hue=col, palette='Set2')
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.show()

    def plot_numerical_distributions(self):
        num_cols, _ = self.check_feature_types()
      
        plot_cols = num_cols 

        n_rows, n_cols, figsize = self._get_grid_params(plot_cols, n_cols=2)
        plt.figure(figsize=figsize)
        plt.suptitle("Continuous Variables Distribution", fontsize=20, fontweight='bold')

        for i, col in enumerate(plot_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=self.data, x=col, kde=True, color='steelblue')
            plt.title(f'Dist of {col}')
            
        plt.tight_layout()
        plt.show()

    def plot_outliers(self):
        num_cols, _ = self.check_feature_types()
        
        n_rows, n_cols, figsize = self._get_grid_params(num_cols, n_cols=2)
        plt.figure(figsize=figsize)
        plt.suptitle("Outlier Analysis (Boxplots)", fontsize=20, fontweight='bold')

        for i, col in enumerate(num_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(data=self.data, x=col, hue=self.target_col, palette='Set3')
            plt.title(f'Boxplot {col}', fontsize=14)
            
        plt.tight_layout()
        plt.show()

    def plot_categorical_by_target(self):
        _, cat_cols = self.check_feature_types()
        plot_cols = [c for c in cat_cols if c != self.target_col and self.data[c].nunique() <= 15]

        n_rows, n_cols, figsize = self._get_grid_params(plot_cols)
        plt.figure(figsize=figsize)
        plt.suptitle(f"Categorical Analysis by {self.target_col}", fontsize=20, fontweight='bold')

        for i, col in enumerate(plot_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.title(f'{col} vs {self.target_col}')
            sns.countplot(
                data=self.data, 
                x=col, 
                hue=self.target_col,
                palette='Set2'
                )
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_numerical_by_target(self):
        num_cols, _ = self.check_feature_types()
        plot_cols = [c for c in num_cols if c != self.target_col]

        n_rows, n_cols, figsize = self._get_grid_params(plot_cols)
        plt.figure(figsize=figsize)
        plt.suptitle(f"Numerical Analysis by {self.target_col}", fontsize=20, fontweight='bold')

        for i, col in enumerate(plot_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(
                data=self.data, 
                x=self.target_col, 
                y=col, 
                hue=self.target_col,
                legend=False,
                palette='Set2'
                )            
            plt.title(f'{col} by {self.target_col}')

        plt.tight_layout()
        plt.show()