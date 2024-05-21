import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class NBASalaryPredictor:
    def __init__(self, filepath):
        self.data = pd.read_excel(filepath)
        self.stat = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%',
                         'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
                         'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%',
                         'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
        self.X = self.data[self.stat]
        self.y = self.data['Salaries']
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        self.predictions = {}
        self.metrics = {}

    def train_models(self):
        for name, model in self.models.items():
            model.fit(self.X, self.y)
            self.predictions[name] = model.predict(self.X)
        self.predictions['Average'] = np.mean(list(self.predictions.values()), axis=0)

    def evaluate_models(self):
        for name, preds in self.predictions.items():
            mse = mean_squared_error(self.y, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y, preds)
            r2 = r2_score(self.y, preds)
            self.metrics[name] = {
                'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2
            }

    def show_stat_importance(self):
        rf_model = self.models['RandomForest']
        stat_importances = rf_model.feature_importances_
        indices = np.argsort(stat_importances)[::-1]
        stat_sorted = [self.stat[i] for i in indices]

        plt.figure(figsize=(10, 6))
        plt.title("Stat Importance - RandomForest")
        plt.bar(range(len(stat_importances)), stat_importances[indices], align="center")
        plt.xticks(range(len(stat_importances)), stat_sorted, rotation=90)
        plt.tight_layout()
        plt.show()

    def show_predictions(self):
        plt.figure(figsize=(10, 6))
        for name, preds in self.predictions.items():
            plt.scatter(self.y, preds, alpha=0.3, label=name)
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=2)
        plt.xlabel("Actual Salaries")
        plt.ylabel("Predicted Salaries")
        plt.title("Predicted vs Actual Salaries")
        plt.legend()
        plt.show()

    def show_errors(self):
        plt.figure(figsize=(10, 6))
        for name, preds in self.predictions.items():
            errors = preds - self.y
            sns.histplot(errors, bins=50, kde=True, label=name)
        plt.xlabel("Prediction Error")
        plt.title("Distribution of Prediction Errors")
        plt.legend()
        plt.show()

    def save_predictions(self, output_filepath):
        for name, preds in self.predictions.items():
            self.data[f'{name}_Predicted_Salary'] = preds
        self.data.to_excel(output_filepath, index=False)

    def print_metrics(self):
        for name, metric in self.metrics.items():
            print(f"\n{name}:")
            for key, value in metric.items():
                print(f"{key}: {value}")

# Usage
predictor = NBASalaryPredictor('D:\\VSCODE\\Nba Salary Analysis\\CombinedNBAStatsSalaries.xlsx')
predictor.train_models()
predictor.evaluate_models()
predictor.show_stat_importance()
predictor.show_predictions()
predictor.show_errors()
predictor.save_predictions('1predicted_salaries_with_three_models&avg.xlsx')
predictor.print_metrics()
