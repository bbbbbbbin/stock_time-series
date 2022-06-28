# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:57:46 2019

@author: BIN
"""

from func import *
import json
from tqdm import tqdm
import random

sns.set(style='ticks', context='poster')

with open("rounda488.json", encoding='utf-8') as f:
    dic = json.load(f)
dic_stock = {}
for d in dic:
    dic_stock[d] = {}
    dic_stock[d]['x'] = np.array(list(dic[d].keys()),dtype='float64')
    dic_stock[d]['y'] = np.array(list(dic[d].values()),dtype='float64')
    dic_stock[d]['data'] = pd.DataFrame(dic_stock[d]['y'],dic_stock[d]['x'])

# print(len(dic_stock))
# print([d for d in dic_stock])
    '''
random.seed(1)
nl = []
for i in range(10):
    nl.append(random.randint(0,111))
    '''
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
dic_s = {}

for d in tqdm(dic_stock):

# name = list(dic_stock.keys())[i]
    name = d
    dic_s[name]={}
    # print(name)
    x = dic_stock[name]['x'][:485]
    y = dic_stock[name]['y'][:485]
    data = dic_stock[name]['data'][:485]
    
    #plt.figure(figsize=(16,9))
    #sns.lineplot(x,y)
    
    da, db = sm.tsa.filters.hpfilter(y)
    # plt.figure(figsize=(16,9))
    # sns.lineplot(x,da)
    # sns.lineplot(x,db)
    # plt.legend(['cycle','trend'])
    
    # print(adfTest(da))
    best_score,best_cfg = chooseModels2(da)
    dic_s[name]['cycle_order'] = best_cfg
    # res = sm.tsa.arma_order_select_ic(da, max_ar=4, max_ma=4, ic=['aic', 'bic'], trend='nc')
    model, model_result = fitModel(da,best_cfg)
    #if model_result.pvalues[0] > 0.05:
    #    best_score,best_cfg = chooseModels2(da,trend='nc')
    #    model, model_result = fitModel(da,best_cfg,'nc')
    # print(best_cfg)
    model_result.summary()
    # plt.figure(figsize=(16,9))
    # sns.lineplot([i for i in range(len(model_result.resid))],model_result.resid)
    acf_pacf_Test(model_result.resid,12)
    cp = forcastInModel(model_result)
    cpo,cpci = forcastOutModel(model_result,3)
    # plt.figure(figsize=(16,9))
    # sns.lineplot(x,da)
    # sns.lineplot(x,cp)
    # plt.legend(['real','predict'])
    
    # print(adfTest(db))
    '''
    temp= pd.DataFrame(db).copy()
    temp['diff'] = temp[temp.columns[0]]
    temp['diff'] = temp['diff'].diff(1)
    plt.figure(figsize=(16,9))
    plt.plot(temp['diff'])
    adfuller(temp['diff'].dropna())
    temp['diff'] = temp['diff'].diff(1)
    plt.figure(figsize=(16,9))
    plt.plot(temp['diff'])
    adfuller(temp['diff'].dropna())
    '''
    dba = np.log(db)
    td, tfv = bestDiff(pd.DataFrame(dba,index=[float(i) for i in range(1,len(dba)+1)],columns=[0]))
    best_score,best_cfg = chooseModels2(td)
    dic_s[name]['trend_order'] = best_cfg
    model, model_result = fitModel(td,best_cfg)
    #if model_result.pvalues[0] > 0.05:
    #    best_score,best_cfg = chooseModels2(da,trend='nc')
    #    model, model_result = fitModel(td,best_cfg,'nc')
    # print(best_cfg)
    model_result.summary()
    # plt.figure(figsize=(16,9))
    # sns.lineplot([i for i in range(len(model_result.resid))],model_result.resid)
    acf_pacf_Test(model_result.resid,12)
    tp = forcastInModel(model_result)
    tpo,tpci = forcastOutModel(model_result,3)
    # plt.figure(figsize=(16,9))
    # sns.lineplot(x[len(tfv):],td)
    # sns.lineplot(x[len(tfv):],tp)
    # plt.legend(['real','predict'])
    
    df = pd.DataFrame(tp,index=list(range(len(tfv)+1,len(tp)+len(tfv)+1)))
    dfp = pd.DataFrame(np.append(tp,tpo),index=list(range(len(tfv)+1,len(tp)+len(tfv)+1+3)))
    tpr = recoverDiff(df, tfv)
    tppr = recoverDiff(dfp, tfv)
    
    tpr = np.exp(tpr[0])
    tppr = np.exp(tppr[0])
    # plt.figure(figsize=(16,9))
    # sns.lineplot(x,db)
    # sns.lineplot(x,tpr[0])
    # sns.lineplot(x,tpr)
    # plt.legend(['real','predict'])
    
    '''
    plt.figure(figsize=(16,9))
    sns.lineplot(x,y)
    sns.lineplot(x,tpr+cp)
    plt.legend(['real','predict'])
    plt.title(name)
    '''
    
    # print(evaluationValue(dic_stock[name]['y'][-3:]/np.mean(dic_stock[name]['y']),(tppr.values[-3:]+cpo)/np.mean(dic_stock[name]['y'])))
    # print(evaluationValue(dic_stock[name]['y'][-3:],tppr.values[-3:]+cpo))
    output = evaluationValue(dic_stock[name]['y'][-3:],tppr.values[-3:]+cpo)
    dic_s[name]['cpo'] = list(cpo)
    dic_s[name]['y_ture'] = list(dic_stock[name]['y'][-3:])
    dic_s[name]['tpo'] = list(tppr.values[-3:])
    dic_s[name]['MSE'] = output['MSE'].values[0]
    dic_s[name]['RMSE'] = output['RMSE'].values[0]
    dic_s[name]['MAE'] = output['MAE'].values[0]

with open('threepointtwo.json','w') as f:
    f.write(json.dumps(dic_s,ensure_ascii=False,cls=MyEncoder,indent=2))
    f.close()
    