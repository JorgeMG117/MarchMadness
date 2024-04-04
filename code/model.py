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
        dtrain = xgb.DMatrix(X, label=y)
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
        self.dtrain = dtrain

        oof_preds = []
        for i in range(self.repeat_cv):
            print(f"Fold repeater {i}")
            preds = np.zeros_like(y, dtype=float)
            kfold = KFold(n_splits=5, shuffle=True, random_state=i)
            for train_index, val_index in kfold.split(X, y):
                dtrain_i = xgb.DMatrix(X.iloc[train_index], label=y.iloc[train_index])
                dval_i = xgb.DMatrix(X.iloc[val_index], label=y.iloc[val_index])
                model = xgb.train(
                    params=self.param,
                    dtrain=dtrain_i,
                    num_boost_round=self.iteration_counts[i],
                    verbose_eval=50
                )
                preds[val_index] = model.predict(dval_i)

            oof_preds.append(np.clip(preds, -30, 30))
        self.oof_preds = oof_preds
        return oof_preds


    def predict_matchup(self, Xsub, sub):
        dtest = xgb.DMatrix(Xsub)
        sub_models = []

        for i in range(self.repeat_cv):
            print(f"Fold repeater {i}")
            sub_models.append(
                xgb.train(
                    params=self.param,
                    dtrain=self.dtrain,
                    num_boost_round=int(self.iteration_counts[i] * 1.05),
                    verbose_eval=50
                )
            )

        sub_preds = []
        for model in sub_models:
            sub_preds.append(model.predict(dtest))

        # Assuming 'spline_model' and further processing to apply here, update accordingly
        sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
        sub[['ID', 'Pred']].to_csv("submission.csv", index=None)

    @staticmethod
    def visualize_performance(oof_preds, y):
        plot_df = pd.DataFrame({"pred": oof_preds[0], "label": np.where(y > 0, 1, 0)})
        plot_df["pred_int"] = plot_df["pred"].astype(int)
        plot_df = plot_df.groupby('pred_int')['label'].mean().reset_index(name='average_win_pct')

        plt.figure()
        plt.plot(plot_df.pred_int, plot_df.average_win_pct)
        plt.show()
