import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class XGBPredictor:
    def __init__(self):
        self.param = {
            'eval_metric': 'mae',
            'booster': 'gbtree',
            'eta': 0.05,
            'subsample': 0.35,
            'colsample_bytree': 0.7,
            'num_parallel_tree': 3,
            'min_child_weight': 40,
            'gamma': 10,
            'max_depth': 3,
            'verbosity': 0
        }
        self.repeat_cv = 3  # Consider setting this to 10 for final runs

    def cauchyobj(self, preds, dtrain):
        labels = dtrain.get_label()
        c = 5000
        x = preds - labels
        grad = x / (x**2/c**2+1)
        hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
        return grad, hess

    def train_model(self, X, y):
        dtrain = xgb.DMatrix(X.values, label=y)
        xgb_cv = []

        for i in range(self.repeat_cv):
            print(f"Fold repeater {i}")
            xgb_cv.append(
                xgb.cv(
                    params=self.param,
                    dtrain=dtrain,
                    obj=self.cauchyobj,
                    num_boost_round=3000,
                    folds=KFold(n_splits=5, shuffle=True, random_state=i),
                    early_stopping_rounds=25,
                    verbose_eval=50
                )
            )

        self.iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
        print("Repeat CV result: ", self.iteration_counts)
        val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
        print("Validation MAE: ", val_mae)
        self.dtrain = dtrain
        #print(xgb_cv)

        # Train final model on the entire dataset
        num_boost_round = int(np.mean(self.iteration_counts))
        print(f"num_boost_rounds={num_boost_round}")
        self.final_model = xgb.train(params=self.param, dtrain=dtrain, num_boost_round=num_boost_round)

        


    def predict_matchup(self, matchup):
        dtest = xgb.DMatrix(matchup.values)
        
        sub_preds = []
        for model in self.final_model:
            sub_preds.append(model.predict(dtest))

        if pd.DataFrame(sub_preds).mean(axis=0).item() > 0.5:
            return True
        else:
            return False
        #sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
        #sub[['ID', 'Pred']].to_csv("submission.csv", index=None)

    @staticmethod
    def visualize_performance(oof_preds, y):
        plot_df = pd.DataFrame({"pred": oof_preds[0], "label": np.where(y > 0, 1, 0)})
        plot_df["pred_int"] = plot_df["pred"].astype(int)
        plot_df = plot_df.groupby('pred_int')['label'].mean().reset_index(name='average_win_pct')

        plt.figure()
        plt.plot(plot_df.pred_int, plot_df.average_win_pct)
        plt.show()
