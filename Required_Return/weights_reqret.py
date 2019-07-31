import numpy as np
import pandas as pd
from numpy.linalg import inv
tbl=pd.read_excel('/Users/nicholesattar/Desktop/Workbook4.xlsx', usecols = (0,1,2), sheet_name='Sample Data')
reqret = 0.036
mean = pd.DataFrame(tbl.mean(),index=tbl.columns)
stdevp = pd.DataFrame(tbl.std(),index=tbl.columns)
means=np.matmul(mean,mean.T)

Assets = pd.DataFrame(np.stack((mean.iloc[:,0],stdevp.iloc[:,0]), axis=0), index = {'means','stdevp'}, columns = list(mean.index.values))
V = (tbl.T).dot(tbl)/1000 - means
A = (tbl.T).dot(tbl)/1000 - means
A[''] = pd.Series(np.ones(len(A.iloc[0]))*-1, index=A.index)
A[' '] = mean*-1
A.loc[''] = np.append([np.ones((len(A.columns)-2))],[0,0])
A.loc[' '] = pd.concat([mean]).T.iloc[0]
A = A.fillna(0)

b = np.append(pd.Series(np.zeros(len(A.iloc[0])-2)),[1,reqret])
weights = pd.DataFrame(np.matmul(inv(A),b)[0:len(np.matmul(inv(A),b))-2],index=mean.index)
ret = (weights*mean).sum()
risk = pd.DataFrame(np.sqrt(np.matmul(np.matmul(weights.T,V),weights)))[0]
lam = pd.Series(np.matmul(inv(A),b)[-2])
mu = pd.Series(np.matmul(inv(A),b)[-1])
rrl=pd.DataFrame([ret,risk,lam,mu],index=('return','risk','lambda','mu'))
MCR = pd.DataFrame(np.matmul(V,weights)/risk.iloc[0],index=tbl.columns)
attribution = pd.DataFrame(MCR*weights/risk)
rrl = rrl.rename(index=str, columns={0: 'greeks'})
weights = weights.rename(index=str, columns={0: 'weights'})
MCR = MCR.rename(index=str, columns={0: 'MCR'})
attribution = attribution.rename(index=str, columns={0: 'attribution'})
print(weights)
print(' ')
print(rrl)
print(' ')
print(MCR)
print(" ")
print(attribution)

