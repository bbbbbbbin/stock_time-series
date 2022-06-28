
# coding: utf-8

# In[1]:


# import import_ipynb


# In[2]:


from func import *
import json
from tqdm import tqdm


# In[3]:
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


# In[4]:


dic_stock = {}
for d in dic:
    dic_stock[d] = {}
    dic_stock[d]['x'] = np.array(list(dic[d].keys()),dtype='float64')
    dic_stock[d]['y'] = np.array(list(dic[d].values()),dtype='float64')
    dic_stock[d]['data'] = pd.DataFrame(dic_stock[d]['y'],dic_stock[d]['x'])


# In[5]:


dic_stock


# In[6]:


len(dic_stock)


# In[30]:


def fit_model(dic,name,x,y,data,adfmethod='hp',modelmethod='bic',maxlag=5,fstep=3,flag='data'):
    
    if not adfTest(y):
        if(adfmethod == 'hp'):
            cycle,trend = hpFilter(y)
            dic[name]['hp'] = {'cycle':list(cycle),'trend':list(trend)}
            fit_model(dic,name,x,cycle,pd.DataFrame(cycle,x),adfmethod='diff',maxlag=5,fstep=fstep,flag='cycle')
            fit_model(dic,name,x,trend,pd.DataFrame(trend,x),adfmethod='diff',maxlag=5,fstep=fstep,flag='trend')
        elif(adfmethod == 'diff'):
            dy,dyf = bestDiff(pd.DataFrame(y))
            y = dy.values
            yf = [i.tolist() for i in dyf]
            yf = list(sum(yf,[]))
            dic[name]['diff'][flag] = {'dy':list(y), 'dy_firstvalue':yf}
        else:
            print('error adf!')
    
    best_score,best_cfg = chooseModels2(y,maxlag,method=modelmethod)
    dic[name]['pq'][flag] = {'best_cfg':best_cfg,'best_score':best_score}
    model, model_result = fitModel(y,best_cfg)
    # dic[name]['model'][flag] = {'model':model, 'model_result':model_result}
    dic[name]['modelfit'][flag] = modelBLQ(model_result,int(len(y)*0.33))
    
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
        pyo,pyciu,pycid = pyo[-fstep:],pyciu[-fstep:],pycid[-fstep:]
    
    dic[name]['predict'][flag] = {'py':list(py),'pyo':list(pyo),'pyciu':list(pyciu),'pycid':list(pycid)}
    
    return dic
    


# In[37]:


fstep = 3
dic = {}
for d in tqdm(dic_stock):
    name = d
    x = dic_stock[name]['x'][:-fstep]
    y = dic_stock[name]['y'][:-fstep]
    # x = dic_stock[name]['x']
    # y = dic_stock[name]['y']
    data = dic_stock[name]['data']
    dic[name] = {}
    dic[name]['data'] = list(y)
    dic[name]['hp'] = None
    dic[name]['diff'] = {}
    dic[name]['pq'] = {}
    dic[name]['model'] = {}
    dic[name]['modelfit'] = {}
    dic[name]['predict'] = {}
    dic[name]['real'] = list(dic_stock[name]['y'][-fstep:])
    dic[name]['Volatility'] = None
    dic[name]['return'] = None
    dic[name]['predict_return'] = None
    dic = fit_model(dic,name,x,y,data,adfmethod='hp',modelmethod='bic',maxlag=5,fstep=fstep,flag='data')
    dic = fit_model(dic,name,x,y,data,adfmethod='diff',modelmethod='bic',maxlag=5,fstep=3,flag='data')


with open('test.json','w') as f:
    f.write(json.dumps(dic,ensure_ascii=False,cls=MyEncoder,indent=2))
    f.close()
    
# In[32]:


plt.figure(figsize=(16,9))
sns.lineplot(x,y)
sns.lineplot(x,dic[name]['predict']['trend']['py']+dic[name]['predict']['cycle']['py'])


# In[33]:


# dic[name]['pq']


# In[34]:


# len(dic[name]['predict']['trend']['pyo'])


# In[35]:


# dic


# In[36]:


plt.figure(figsize=(16,9))
sns.lineplot([i for i in range(fstep)],dic[name]['real'])
sns.lineplot([i for i in range(fstep)],dic[name]['predict']['trend']['pyo']+dic[name]['predict']['cycle']['pyo'])
plt.fill_between([i for i in range(fstep)],dic[name]['predict']['trend']['pyciu']+dic[name]['predict']['cycle']['pyciu'],dic[name]['predict']['trend']['pycid']+dic[name]['predict']['cycle']['pycid'],alpha=0.5)


# In[ ]:


evaluationValue(dic[name]['real'],dic[name]['predict']['trend']['pyo']+dic[name]['predict']['cycle']['pyo'])

