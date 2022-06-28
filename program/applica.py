# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:41:01 2019

@author: BIN
"""

from func import *
import json
from tqdm import tqdm

sns.set(style='ticks', context='poster')

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

with open("rounda488.json", encoding='utf-8') as f:
    dic = json.load(f)
    
dl = ['stock103528', 'stock100042', 'stock102820', 'stock101651', 'stock100591'\
      , 'stock102273', 'stock101712', 'stock100683', 'stock102926', 'stock100152'\
      , 'stock101758', 'stock103158', 'stock102520', 'stock100497', 'stock101766',\
      'stock103137', 'stock101003', 'stock102156']

dic_stock={}
for d in dic:
    if d not in dl:
        dic_stock[d] = {}
        dic_stock[d]['x'] = np.array(list(dic[d].keys()),dtype='float64')
        dic_stock[d]['y'] = np.array(list(dic[d].values()),dtype='float64')
        dic_stock[d]['data'] = pd.DataFrame(dic_stock[d]['y'],dic_stock[d]['x'])


def fit_model(dic,name,x,y,data,adfmethod='hp',modelmethod='bic',maxlag=5,fstep=3,flag='data',flag2='hp'):
    
    if not adfTest(y):
        if(adfmethod == 'hp'):
            cycle,trend = hpFilter(y)
            dic[name][flag2+'_method']['hp'] = {'cycle':list(cycle),'trend':list(trend)}
            fit_model(dic,name,x,cycle,pd.DataFrame(cycle,x),adfmethod='diff',maxlag=5,fstep=fstep,flag='cycle',flag2='hp')
            fit_model(dic,name,x,trend,pd.DataFrame(trend,x),adfmethod='diff',maxlag=5,fstep=fstep,flag='trend',flag2='hp')
        elif(adfmethod == 'diff'):
            if flag=='cycle' or flag=='trend':
                y = np.log(y)
            dy,dyf = bestDiff(pd.DataFrame(y,index=[float(i) for i in range(1,len(y)+1)],columns=[0]))
            y = dy.values
            yf = [i.tolist() for i in dyf]
            yf = list(sum(yf,[]))
            dic[name][flag2+'_method']['diff'][flag] = {'dy':list(y), 'dy_firstvalue':yf}
        else:
            print('error adf!')
    

    best_score,best_cfg = chooseModels2(y,maxlag,method=modelmethod)
    dic[name][flag2+'_method']['pq'][flag] = {'best_cfg':best_cfg,'best_score':best_score}
    model, model_result = fitModel(y,best_cfg)
    # dic[name]['model'][flag] = {'model':model, 'model_result':model_result}
    dic[name][flag2+'_method']['modelfit'][flag] = modelBLQ(model_result,sum(best_cfg))
    
    py = forcastInModel(model_result)
    pyo,pyci = forcastOutModel(model_result,fstep)
    pyciu = pyci[:,0][-fstep:]
    pycid = pyci[:,1][-fstep:]
    
    if 'dy' in vars():
        df = pd.DataFrame(py,index=list(range(len(dyf)+1,len(py)+len(dyf)+1)))
        dfp = pd.DataFrame(np.append(py,pyo),index=list(range(len(dyf)+1,len(py)+len(dyf)+1+fstep)))
        dfu = pd.DataFrame(np.append(py,pyci[:,0]),index=list(range(len(dyf)+1,len(py)+len(dyf)+1+fstep)))
        dfd = pd.DataFrame(np.append(py,pyci[:,1]),index=list(range(len(dyf)+1,len(py)+len(dyf)+1+fstep)))
        py = recoverDiff(df, dyf)
        pyo = recoverDiff(dfp, dyf)
        pyciu = recoverDiff(dfu, dyf)
        pycid = recoverDiff(dfd, dyf)
        py,pyo,pyciu,pycid = py[0].values,pyo[0].values,pyciu[0].values,pycid[0].values
        if flag=='cycle' or flag=='trend':
                py,pyo,pyciu,pycid = np.exp(py),np.exp(pyo),np.exp(pyciu),np.exp(pycid)
        pyo,pyciu,pycid = pyo[-fstep:],pyciu[-fstep:],pycid[-fstep:]
    dic[name][flag2+'_method']['predict'][flag] = {'py':list(py),'pyo':list(pyo),'pyciu':list(pyciu),'pycid':list(pycid)}
    
    return dic

dic = {}
for d in tqdm(dic_stock):
    name = d
    x = dic_stock[name]['x']
    y = dic_stock[name]['y']
    data = dic_stock[name]['data']
    
    dic[name]={}
    dic[name]['hp_method']={}
    dic[name]['diff_method']={}
    dic[name]['data'] = list(y)
    dic[name]['hp_method']['hp'] = None
    dic[name]['hp_method']['diff'] = {}
    dic[name]['diff_method']['diff'] = {}
    dic[name]['hp_method']['pq'] = {}
    dic[name]['diff_method']['pq'] = {}
    dic[name]['hp_method']['modelfit'] = {}
    dic[name]['diff_method']['modelfit'] = {}
    dic[name]['hp_method']['predict'] = {}
    dic[name]['diff_method']['predict'] = {}
    dic[name]['Volatility'] = None
    dic[name]['hp_method']['Return'] = None
    dic[name]['diff_method']['Return'] = None
    
    fstep = 3
    dic = fit_model(dic,name,x,y,data,adfmethod='hp',modelmethod='bic',maxlag=5,fstep=fstep,flag='data',flag2='hp')
    dic = fit_model(dic,name,x,y,data,adfmethod='diff',modelmethod='bic',maxlag=5,fstep=fstep,flag='data',flag2='diff')
    
    r = []
    for i in range(1,len(y)):
        ri = np.log(y[i]/y[i-1])
        r.append(ri)
    dic[name]['Volatility'] = np.sqrt(sum([(ri-np.mean(r))*(ri-np.mean(r)) for ri in r])/(len(y)-1))
    
    dic[name]['hp_method']['Return'] = (dic[name]['hp_method']['predict']['cycle']['pyo'][-1] + dic[name]['hp_method']['predict']['trend']['pyo'][-1])/dic[name]['data'][-1] - 1
    dic[name]['diff_method']['Return'] = dic[name]['diff_method']['predict']['data']['pyo'][-1]/dic[name]['data'][-1] - 1

with open('threepointthree.json','w') as f:
    f.write(json.dumps(dic,ensure_ascii=False,cls=MyEncoder,indent=2))
    f.close()