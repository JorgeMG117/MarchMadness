import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


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

        # out-of-fold prediction
        oof_preds = []
        for i in range(self.repeat_cv):
            print(f"Fold repeater {i}")
            preds = y.copy()
            preds = preds.astype(float)
            kfold = KFold(n_splits = 5, shuffle = True, random_state = i)    
            for train_index, val_index in kfold.split(X,y):
                dtrain_i = xgb.DMatrix(X.iloc[train_index], label = y.iloc[train_index])
                dval_i = xgb.DMatrix(X.iloc[val_index], label = y.iloc[val_index])  
                model = xgb.train(
                    params = self.param,
                    dtrain = dtrain_i,
                    num_boost_round = self.iteration_counts[i],
                    verbose_eval = 50
                )
                preds[val_index] = model.predict(dval_i)
            oof_preds.append(np.clip(preds,-30,30))

        #self.visualize_performance(oof_preds, y)


        # Spline interpolation
        val_cv = []
        spline_model = []

        for i in range(self.repeat_cv):
            dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
            dat = sorted(dat, key = lambda x: x[0])
            datdict = {}
            for k in range(len(dat)):
                datdict[dat[k][0]]= dat[k][1]
            spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
            spline_fit = spline_model[i](oof_preds[i])
            spline_fit = np.clip(spline_fit,0.025,0.975)
            """
            spline_fit[(tourney_data.T1_seed==1) & (tourney_data.T2_seed==16) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
            spline_fit[(tourney_data.T1_seed==2) & (tourney_data.T2_seed==15) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
            spline_fit[(tourney_data.T1_seed==3) & (tourney_data.T2_seed==14) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
            spline_fit[(tourney_data.T1_seed==4) & (tourney_data.T2_seed==13) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
            spline_fit[(tourney_data.T1_seed==16) & (tourney_data.T2_seed==1) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
            spline_fit[(tourney_data.T1_seed==15) & (tourney_data.T2_seed==2) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
            spline_fit[(tourney_data.T1_seed==14) & (tourney_data.T2_seed==3) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
            spline_fit[(tourney_data.T1_seed==13) & (tourney_data.T2_seed==4) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
            """
            #val_cv.append(pd.DataFrame({"y":np.where(y>0,1,0), "pred":spline_fit, "season":tourney_data.Season}))
            print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 
            
        #val_cv = pd.concat(val_cv)
        #val_cv.groupby('season').apply(lambda x: log_loss(x.y, x.pred))
        #print(val_cv)

        plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0), "spline":spline_model[0](oof_preds[0])})
        plot_df["pred_int"] = (plot_df["pred"]).astype(int)
        plot_df = plot_df.groupby('pred_int')[['spline', 'label']].mean().reset_index()

        plt.figure()
        plt.plot(plot_df.pred_int,plot_df.spline)
        plt.plot(plot_df.pred_int,plot_df.label)
        #plt.show()


        # Train final model on the entire dataset
        #num_boost_round = int(np.mean(self.iteration_counts))
        #print(f"num_boost_rounds={num_boost_round}")
        #self.final_model = xgb.train(params=self.param, dtrain=dtrain, num_boost_round=num_boost_round)
        sub_models = []
        for i in range(self.repeat_cv):
            print(f"Fold repeater {i}")
            sub_models.append(
                xgb.train(
                params = self.param,
                dtrain = dtrain,
                num_boost_round = int(self.iteration_counts[i] * 1.05),
                verbose_eval = 50
                )
            )

        self.sub_models = sub_models
        self.spline_model = spline_model

        


    def predict_matchup(self, matchup):
        dtest = xgb.DMatrix(matchup.values)
        """
        sub_preds = []
        for model in self.final_model:
            sub_preds.append(model.predict(dtest))
        """
        sub_preds = []
        for i in range(self.repeat_cv):
            sub_preds.append(np.clip(self.spline_model[i](np.clip(self.sub_models[i].predict(dtest),-30,30)),0.025,0.975))

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
