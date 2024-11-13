import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error


class Validation:
    
    def __init__(self, parent_log=None):
        if parent_log:
            self.log = parent_log

    @staticmethod
    def r2(y_true, y_pred): #1
        return r2_score(y_true, y_pred)

    @staticmethod
    def bias(y_true, y_pred): #2
        return np.mean(np.asarray(y_pred) - np.asarray(y_true))

    @staticmethod
    def rmse(y_true, y_pred): #3
        # return np.sqrt(((y_pred - y_true) ** 2).mean())
        return np.sqrt(mean_squared_error( y_true, y_pred ))

    @staticmethod
    def rrmse(y_true, y_pred): #4
        # https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/
        """
        Model accuracy is:
        RRMSE < 10% (Excellent)
        RRMSE is between 10% and 20% (Good)
        RRMSE is between 20% and 30% (Fair)
        RRMSE > 30% (Poor)
        """
        num = np.sum(np.square(np.asarray(y_true) - np.asarray(y_pred)))
        den = np.sum(np.square(y_pred))
        squared_error = num/den
        rrmse = np.sqrt(squared_error)
        return rrmse

    # implementation of NRMSE with standard deviation
    def nrmse(self, y_true, y_pred): #5
        """
        NRMSE is a good measure when you want to compare the models of different dependent variables or 
        when the dependent variables are modified (log-transformed or standardized). It overcomes the 
        scale-dependency and eases comparison between models of different scales or even between datasets.
        """
        local_rmse = self.rmse(y_true, y_pred)
        nrmse = local_rmse / np.std(y_pred)
        return nrmse

    @staticmethod
    def rmsle(y_true, y_pred): #6
        return np.sqrt(mean_squared_log_error( y_true, y_pred ))

    # TODO: Add WMAPE

    @staticmethod
    def mape(y_true, y_pred, drop_zero=True): #7
        if 0 in y_true and drop_zero:
            print('Zero values present in Y-truth set. MAPE will attempt to be performed without them.')
            try:
                df = pd.DataFrame(data={'y_true':y_true,'y_pred':y_pred})
                zero_tot = len(df.loc[df['y_true'] == 0])
                print(f'Removing {zero_tot}/{len(df)} values.')
                df = df[df['y_true'] != 0].copy()
                return mean_absolute_percentage_error(df['y_true'], df['y_pred'])
            except:
                e = sys.exc_info()[0]
                print(e)
        return mean_absolute_percentage_error(y_true, y_pred)

    @staticmethod
    def corr(y_true, y_pred, m='spearman'):  # 8 #9 #10
        # Test for same DataFrame
        if isinstance(y_true, pd.Series) and isinstance(y_true, pd.Series):
            # Compute pairwise correlation of columns, excluding NA/null values.
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
            return y_true.corr(y_pred, method=m)
        else:
            # Test for other dtypes and convert to DF
            try:
                df = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
                return df['y_true'].corr(df['y_pred'], method=m)
            except:
                e = sys.exc_info()[0]
                print(e)

    def get_stats(self, y_true, y_pred):
        res = {}
        res['R2'] = round(self.r2(y_true, y_pred), 2)
        res['BIAS'] = round(self.bias(y_true, y_pred), 2)
        res['MAPE'] = round(self.mape(y_true, y_pred), 2)
        res['RMSE'] = round(self.rmse(y_true, y_pred), 2)
        res['RRMSE'] = round(self.rrmse(y_true, y_pred), 2)
        res['NRMSE'] = round(self.nrmse(y_true, y_pred), 2)
        #res['RMSLE'] = round(self.rmsle(y_true, y_pred), 2)
        res['Spearman'] = round(self.corr(y_true, y_pred), 2)  # default m = spearman
        res['Pearson'] = round(self.corr(y_true, y_pred, m='pearson'), 2)
        res['Kendall'] = round(self.corr(y_true, y_pred, m='kendall'), 2)
        return res
