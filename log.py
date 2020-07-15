# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:32:47 2020
@author: Robert Cunningham
"""

import pandas as pd
from datetime import datetime as dt
import os
log = pd.read_csv('logs/log.txt',header=None,engine='python',sep='<--')
log[0] = log[0].str.replace('Line ','')
log[0] = log[0].str.split(' ',1,expand=True).set_index(0)
log.index = log.index.astype(int) - 1
log = log.rename(columns={1:'log',2:'log1'})

out = pd.read_csv('out/out.csv')
log = out.join(log)

cols = ['Item Type','Product Code/SKU','Category','Brand Name','Product Name','log']
log[cols].to_csv('logs/log.csv',index=False)
end = dt.now()
os.rename('logs/log.txt',f'logs/archive/log_{str(end.date())} {end.hour}-{end.minute}-{end.second}.txt')




