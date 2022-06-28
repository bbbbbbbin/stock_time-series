
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller     # 平稳性检验
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
import warnings
from pywt import wavedec,waverec
warnings.filterwarnings("ignore")


# In[2]:


# 平稳性检验
def adfTest(data):
    dftest = adfuller(data,autolag='BIC')
    if dftest[1] < 0.001:
        return True
    else:
        return False


# In[3]:


# ACF和PACF
def acf_pacf_Test(data,n):
    acf,q,p = sm.tsa.acf(data,nlags=n,qstat=True)
    pacf = sm.tsa.acf(data,nlags=n)
    out = np.c_[range(1,n+1), acf[1:], pacf[1:], q, p]
    output=pd.DataFrame(out, columns=['lag', "AC", "PAC", "Q", "P-value"])
    return output


# In[4]:


# 返回各回归评估参数值MSE,RMSE,MAE,R2
def evaluationValue(data_true,data_predict):
    mse = np.sum((data_predict-data_true)**2) / len(data_true)
    rmse = np.sqrt(mse)
    mae = np.sum(np.absolute(data_predict-data_true)) / len(data_true)
    r2 = 1- mse / np.var(data_true)
    out = np.c_[1, mse, rmse, mae, r2]
    output=pd.DataFrame(out, columns=['index', "MSE", "RMSE", "MAE", "R2"])
    return output


# In[5]:


# 差分处理,默认最大为5阶
def bestDiff(df, maxdiff = 6):
    temp = df.copy()
    first_values = []
    for i in range(0, maxdiff):
        if i == 0:
            temp['diff'] = temp[temp.columns[0]]
        else:
            first_values.append(pd.Series([temp['diff'][i]],index=[temp['diff'].index[0]]))
            temp['diff'] = temp['diff'].diff(1)
            temp = temp.dropna() #差分后，前几行的数据会变成nan，所以删掉
            # print(temp['diff'],'\n')
        if adfTest(temp['diff']):
            bestdiff = i
            return temp['diff'],first_values
        else:
            continue
    return temp['diff'],first_values


# In[6]:


# 差分恢复
def recoverDiff(df_diff,first_values):
    df_restored = df_diff
    for first in reversed(first_values):
        df_restored = first.append(df_restored).cumsum()
    return df_restored


# In[7]:


# HP分解
def hpFilter(data,l=100):
    cycles, trend = sm.tsa.filters.hpfilter(data,l)
    return cycles,trend


# In[8]:


# DW检验
def evaluationDW(resid):
    return sm.stats.durbin_watson(resid)


# In[9]:


# 小波分解
def waveletFilter(data,level,func='db4'):
    coeffs = wavedec(data, func, level=level)
    return coeffs


# In[10]:


# 小波恢复
def recoverWavelet(coeffs,func='db4'):
    data = waverec(coeffs, func)
    return data


# In[13]:


# 模型评估,滚动预测
def evaluationModle(data,order,train_size=-1):
    if train_size == -1:    
        train_size = int(len(data) * 0.66)
    train, test = data[0:train_size], data[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARMA(history, order=order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat[0])
        history.append(test[t])
    error = evaluationValue(test, predictions)
    # print(predictions)
    sns.lineplot(np.array(list(range(len(predictions)))),np.array(predictions),color='r')
    sns.lineplot(np.array(list(range(len(test)))),test,color='b')
    plt.show()
    return error,predictions


# In[14]:


# 模型PQ选择，方法1（使用评估参数）
def chooseModels1(data, maxlag=5,method='MSE'):
    best_score, best_cfg = float("inf"), None
    for p in np.arange(maxlag):
        for q in np.arange(maxlag):
            order = (p,q)
            try:
                error,predictions = evaluationModle(data, order)
                # print(order, error[method].values)
                if error[method].values < best_score:
                    best_score, best_cfg = error[method].values, order
            except:
                # print(order,'error')
                continue
    return best_score,best_cfg


# In[15]:


# 模型PQ选择，方法2（使用aic,bic,hqic）
def chooseModels2(data, maxlag=5,method='bic',trend='c'):
    best_score, best_cfg = float("inf"), None
    for p in np.arange(maxlag):
        for q in np.arange(maxlag):
            order = (p,q)
            model = ARMA(data, order=order)
            try:
                results_ARMA = model.fit(disp=0,trend=trend)
                if method == 'aic':
                    score = results_ARMA.aic
                elif method == 'bic':
                    score = results_ARMA.bic
                elif method == 'hqic':
                    score = results_ARMA.hqic
                print(order, score)
                if score < best_score:
                    best_score, best_cfg = score, order
            except:
                print(order, 'error')
                continue
    return best_score,best_cfg


# In[17]:


# 模型拟合
def fitModel(data,order,trend='c'):
    model = ARMA(data,order=order)
    model_result = model.fit(disp=0,trend=trend)
    return model,model_result


# In[18]:


# 模型检验
def modelBLQ(model_result,n):
    output = acf_pacf_Test(model_result.resid,n)['P-value']
    for i in range(len(output)):
        if output[i] < 0.05:
            return False
    return True


# In[19]:


# 向内模型预测
def forcastInModel(model_result):
    train_predict = model_result.predict()
    return train_predict


# In[20]:


# 向外模型预测
def forcastOutModel(model_result,step):
    train_predict,b,train_predict_conf_int = model_result.forecast(step)
    return train_predict,train_predict_conf_int

