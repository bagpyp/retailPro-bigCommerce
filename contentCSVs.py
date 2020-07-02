# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:19:33 2020

@author: Web
"""
import pandas as pd
pd.options.display.max_columns = 80
pd.options.display.max_rows = 150
pd.options.display.width = 150
df = pd.read_pickle('forBCL.pkl')


bcl = pd.read_pickle('../../barcodeLookupAPI/allOfBCL.pkl')\
    .sort_index()\
    .drop(columns='UPC')
        
bcl.index = bcl.index.astype(int)

# img = pd.read_csv('images.csv').set_index('sku').sort_index()
# desc = pd.read_csv('descriptions.csv').set_index('sku').sort_index()

for X in [bcl,desc,img]:
    X.index = X.index.astype(str).str.zfill(5)
del X

"""
img.notna().sum().sum()
Out[69]: 18935
"""
#%%
goods = pd.DataFrame(index = df.index,\
                     columns = img.columns.tolist()\
                         +desc.columns.tolist())
    
goods.update(img)

goods.update(desc)

goods.update(bcl.rename(columns=lambda x: x.replace('_','')))

goods.description.update(goods.short_description.rename('description'))

goods.iloc[:,:5].to_csv('pics.csv')
goods.iloc[:,-2:].to_csv('descriptions.csv')
