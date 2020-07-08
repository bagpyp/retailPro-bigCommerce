# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:39:10 2020

@author: Web
"""

print('this process takes roughly two minutes, please be patient.')
from datetime import datetime as dt
# get it!
import pandas as pd

import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
import os
pd.options.display.max_columns = 80
pd.options.display.max_rows = 150
pd.options.display.width = 150


# on server cmd prompt, issue: 
# E:\ECM> ecmproc -out -show -stid:001001A
# takes roughly 1.5min


"""INVENTORY"""

#%%
inventorys = []

for file in glob(r'T:\ECM\Polling\001001A\OUT\Inventory*'):
    
    tree = ET.parse(file)
    root = tree.getroot()
    
    invns = root.findall('./INVENTORYS/INVENTORY')



    for j in range(len(invns)):
        invnTags = invns[j].findall('.*')
        pTags = invnTags[2].findall('.*')[2].findall('.*')
        qTags = invnTags[2].findall('.*')[3].findall('.*')
        inventorys.append(invnTags[0].attrib)
        for k in range(1,len(invnTags)):
            inventorys[-1].update(invnTags[k].attrib)
        for l in range(len(pTags)):
            p = pTags[l].attrib
            inventorys[-1].update({k+'_'+f"{p['price_lvl']}":v for k,v in p.items()})
        for m in range(len(qTags)):
            q = qTags[m].attrib
            inventorys[-1].update({k+'_'+f"{q['store_no']}":v for k,v in q.items()})

# del tree, root, file,j,k,l,m

#%%
df = pd.DataFrame(inventorys).replace('',np.nan).dropna(axis=1, how='all')
for col in df.columns:
    if df[col].nunique() == 1:
        df.drop(col,inplace=True,axis=1)
        
# del col

# del p, q, invns, inventorys, pTags, qTags, invnTags


"""CODES"""

#%%

#vender
tree = ET.parse('T:/ECM/Polling/001001A/OUT/Vendor.xml')
root = tree.getroot()

vendors = []

vens = root.findall('./VENDORS/VENDOR')
for i in range(len(vens)):
    vendors.append(vens[i].attrib)

v = pd.DataFrame(vendors).iloc[:,[0,2]].rename(columns={'vend_name':'BRAND'})

# del vens, vendors
#%%

#cats
tree = ET.parse('T:/ECM/Polling/001001A/OUT/DCS.xml')
root = tree.getroot()

categories = []

cats = root.findall('./DCSS/DCS')
for i in range(len(cats)):
    categories.append(cats[i].attrib)

c = pd.DataFrame(categories).iloc[:,[0,2,3,4]]
# format text
c.iloc[:,1:]=c.iloc[:,1:].applymap(str.title)
#drop 5- and 7- length strings
c = c[(c.dcs_code.str.len()==6)|(c.dcs_code.str.len()==9)]\
    .reset_index(drop=True)
#map = dict(v.values)  
    
# del cats, categories

def D(x):
    return x[:3]
def C(x):
    return x[3:6]
def S(x):
    if len(x) == 9:
        return x[6:9]
    else: return ''

c = c.join(pd.Series(c.dcs_code.values).transform([D,C,S]))
c['CAT'] = c.iloc[:,[1,2,3]].apply(\
    lambda x: '/'.join(x.dropna().values.tolist()), axis=1)\
    .apply(lambda x: x[:-1] if x[-1]=='/' else x)
c = c.iloc[:,[0,4,5,6,7]]
# del D, C, S

#%%

df = pd.merge(df, c, on='dcs_code', how='left', sort=False)
df = pd.merge(df, v, on='vend_code', how='left', sort=False)

# del c, v

names = {'item_no':'sku','local_upc':'UPC','alu':'alu','CAT':'CAT',
          'BRAND':'BRAND','description1':'name','description2':'year',
          'attr':'color','siz':'size','qty_0':'qty0','qty_1':'qty1',
          'qty_250':'qty','cost':'cost','price_1':'pSale',
          'price_2':'pMAP','price_3':'pMSRP','price_4':'pAmazon',
          'price_5':'pSWAP',
          'created_date':'fCreated','fst_rcvd_date':'fRcvd',
          'lst_rcvd_date':'lRcvd','lst_sold_date':'lSold','modified_date':'lEdit',
          'vend_code':'VC','dcs_code':'DCS','D':'D',
          'C':'C','S':'S','style_sid':'ssid','item_sid':'isid','upc':'UPC2'}
df.rename(columns=names, inplace=True)
df = df[list(names.values())]

# del names

""" THIS IS WHERE THE PICKLE'S AT!"""
df.to_pickle('fromECM.pkl')

#%%

df = pd.read_pickle('fromECM.pkl')

"""EDITING"""

#dtypes
for i in range(9,18): df.iloc[:,i] = pd.to_numeric(df.iloc[:,i])
for i in range(18,23): df.iloc[:,i] = pd.to_datetime(df.iloc[:,i])
df.lEdit = df.lEdit.dt.tz_localize(None)

# del i

#filtering by UPC, printing csv of discarded
df = df[df.DCS.str.match(r'^((?!USD|REN).)*$')]

#%%

df = df[(df.UPC.str.len().isin([11,12,13]))&(df.UPC2.notnull())]
df.drop('UPC2', axis=1, inplace=True)

#sort by sku
df.sku=df.sku.astype(int)
df = df.sort_values(by='sku')
df.sku = df.sku.astype(str).str.zfill(5)

df.reset_index(drop=True, inplace=True)
df.set_index('sku', inplace=True)
# df.to_pickle('forBCL.pkl')
# #%%

# """ PICKING UP PICS AND DESCS"""

# df = pd.read_pickle('forBCL.pkl')
img = pd.read_csv('images.csv')
desc = pd.read_csv('descriptions.csv')
def f(df):
    df = df.sort_values(by='sku')
    df.sku = df.sku.astype(str).str.zfill(5)
    df.set_index('sku',inplace=True)
    return df
img = f(img)
desc=f(desc)

df= df.join(img, how='left').join(desc, how='left')

# del img, desc

df.to_pickle('df.pkl')

# del df


#%%
"""BIG COMMERCE"""
dCAT = {\
'Misc':'Misc',
'Service/Ski/Tune':'Service',
'Service/Snowboard/Wax':'Service',
'Service/Ski/Polepart':'Service',
'Service/Ski/Bindngpart':'Service',
'Service/Snowboard/Tune':'Service',
'Service/Ski/Brakes':'Service',
'Service/Ski/Mount':'Service',
'Service/Snowboard/Mount':'Service',
'Service/Ski/Bootfit':'Service',
'Service/Snowboard/Bindngpart':'Service',
'Service/Snowboard/Boot Part':'Service',
'Service/Ski/Boot Part':'Service',
'Service/Ski/Wax':'Service',
'Service/Ski/BindngPart':'Service',
'Disc Golf/Bag':'Misc',
'Electronic/Audio':'Misc',
'Electronic/Camera/Accessory':'Misc',
'Eyewear/Goggles/Accessory':'Gear/Goggles/Accessory',
'Eyewear/Goggles/Moto':'Gear/Goggles/Accessory',
'Eyewear/Goggles/Rep. Lens':'Gear/Goggles/Accessory',
'Eyewear/Goggles/Unisex':'Gear/Goggles/Adult',
'Eyewear/Goggles/Womens':'Gear/Goggles/Adult',
'Eyewear/Goggles/Youth':'Gear/Goggles/Youth',
'Eyewear/Sunglasses/Accessory':'Lifestyle/Accessory',
'Eyewear/Sunglasses/Unisex':'Lifestyle/Accessory',
'Eyewear/Sunglasses/Womens':'Lifestyle/Accessory',
'FBA/Bindings/Womens':'Misc',
'FBA/Board/Youth':'Misc',
'Headwear/Beanie':'Gear/Headwear/Beanie',
'Headwear/Facemask':'Gear/Headwear/Facemask',
'Headwear/Hat':'Lifestyle/Accessory',
'Hike/Pack/Accessory':'Lifestyle/Accessory',
'Hike/Pack/Hydration':'Lifestyle/Accessory',
'Hike/Pack/Map/Book':'Lifestyle/Accessory',
'Hike/Pack/Mens/Shoes':'Lifestyle/Men/Shoes',
'Hike/Pack/Snowboard/Shoes':'Lifestyle/Men/Shoes',
'Hike/Pack/Womens/Shoes':'Lifestyle/Women/Shoes',
'Kayak/Accessory':'Watersport/Kayak',
'Lifejacket/Neoprene/Dog':'Watersport/Life Jackets/Dog',
'Lifejacket/Neoprene/Men':'Watersport/Life Jackets/Adult',
'Lifejacket/Neoprene/Womens':'Watersport/Life Jackets/Adult',
'Lifejacket/Neoprene/Youth':'Watersport/Life Jackets/Youth',
'Lifejacket/Nylon/Men':'Watersport/Life Jackets/Adult',
'Lifejacket/Nylon/Womens':'Watersport/Life Jackets/Adult',
'Lifejacket/Nylon/Youth':'Watersport/Life Jackets/Youth',
'Mens/Baselayer/Bottom':'Gear/Base Layer/Men',
'Mens/Baselayer/Suit':'Gear/Base Layer/Men',
'Mens/Baselayer/Top':'Gear/Base Layer/Men',
'Mens/Lifestyle/Accessory':'Lifestyle/Men/Accessory',
'Mens/Lifestyle/Bag':'Lifestyle/Men/Accessory',
'Mens/Lifestyle/Jacket':'Lifestyle/Men/Jacket',
'Mens/Lifestyle/Pants':'Lifestyle/Men/Pants',
'Mens/Lifestyle/Shoes':'Lifestyle/Men/Shoes',
'Mens/Lifestyle/Shorts':'Lifestyle/Men/Shorts',
'Mens/Lifestyle/Top':'Lifestyle/Men/Tops',
'Mens/Midlayer':'Gear/Midlayer/Men',
'Mens/Outerwear/Gloves':'Gear/Gloves/Adult',
'Mens/Outerwear/Jackets':'Gear/Jacket/Men',
'Mens/Outerwear/Mittens':'Gear/Gloves/Adult',
'Mens/Outerwear/Pants':'Gear/Pants/Men',
'Mens/Outerwear/Suit':'Gear/Outerwear',
'Mens/Swimwear/Shorts':'Lifestyle/Men/Shorts',
'Race/Night':'Misc',
'Safety/Avalanche/Probe':'Misc',
'Safety/Avalanche/Shovel':'Misc',
'Safety/Avalanche/Tranceiver':'Misc',
'Safety/Helmet/Skate':'Skate/Helmets',
'Safety/Helmet/Ski':'Ski/Helmets',
'Safety/Helmet/Wakeboard':'Watersport/Wakeboard/Accessory',
'Safety/Pad/Skate':'Skate/Accessory',
'Safety/Pad/Snow':'Snowboard/Accessory',
'Safety/Race/Ski':'Ski/Accessory',
'Skateboard/Accessory':'Skate/Accessory',
'Skateboard/Bearings':'Skate/Bearings',
'Skateboard/Complete':'Skate/Complete',
'Skateboard/Complete/Street':'Skate/Complete',
'Skateboard/Completes/Long Board':'Skate/Complete',
'Skateboard/Deck/Street':'Skate/Decks',
'Skateboard/Griptape':'Skate/Accessory',
'Skateboard/Hardware':'Skate/Accessory',
'Skateboard/Shoes/Mens':'Skate/Shoes/Men',
'Skateboard/Shoes/Womens':'Skate/Shoes/Women',
'Skateboard/Trucks/Street':'Skate/Trucks',
'Skateboard/Wheels/Longboard':'Skate/Wheels',
'Skateboard/Wheels/Street':'Skate/Wheels',
'Ski/Accessory':'Ski/Accessory',
'Ski/Accessory/Insoles':'Ski/Accessory',
'Ski/Bags/Backpack':'Ski/Accessory',
'Ski/Bags/Boot':'Ski/Accessory',
'Ski/Bags/Gear':'Ski/Accessory',
'Ski/Bags/Ski':'Ski/Accessory',
'Ski/Bindings/Mens':'Ski/Bindings/Adult',
'Ski/Bindings/Womens':'Ski/Bindings/Adult',
'Ski/Bindings/Youth':'Ski/Bindings/Youth',
'Ski/Boots/Accessory':'Ski/Accessory',
'Ski/Boots/Liner':'Ski/Accessory',
'Ski/Boots/Mens':'Ski/Boots/Adult',
'Ski/Boots/Parts':'Ski/Boots/Parts',
'Ski/Boots/Womens':'Ski/Boots/Adult',
'Ski/Boots/Youth':'Ski/Boots/Youth',
'Ski/Poles/Accessory':'Ski/Accessory',
'Ski/Poles/Adult':'Ski/Poles/Adult',
'Ski/Poles/Baskets':'Ski/Accessory',
'Ski/Poles/Youth':'Ski/Poles/Youth',
'Ski/Skis/Mens':'Ski/Skis/Adult',
'Ski/Skis/Womens':'Ski/Skis/Adult',
'Ski/Skis/Youth':'Ski/Skis/Youth',
'Ski/Socks/Adult':'Ski/Socks/Adult',
'Ski/Socks/Youth':'Ski/Socks/Youth',
'Ski/Tune/Wax':'Ski/Accessory',
'Ski/Tuning/Tool':'Ski/Accessory',
'Ski/X-Country/Bindings':'Ski/Accessory',
'Ski/X-Country/Boots':'Ski/Accessory',
'Ski/X-Country/Skis':'Ski/Accessory',
'Snowboard/Accessorie/StompPads':'Snowboard/Accessory',
'Snowboard/Accessory':'Snowboard/Accessory',
'Snowboard/Bags/Backpack':'Snowboard/Bags/Backpack',
'Snowboard/Bags/Board Bag':'Snowboard/Bags/Board',
'Snowboard/Bags/Gear':'Snowboard/Bags/Gear',
'Snowboard/Bags/Travel':'Snowboard/Bags/Travel',
'Snowboard/Bags/Wheel':'Snowboard/Bags/Wheel',
'Snowboard/Bindings/Unisex':'Snowboard/Bindings/Adult',
'Snowboard/Bindings/Women':'Snowboard/Bindings/Adult',
'Snowboard/Bindings/Youth':'Snowboard/Bindings/Youth',
'Snowboard/Board/Mens':'Snowboard/Board/Adult',
'Snowboard/Board/Womens':'Snowboard/Board/Adult',
'Snowboard/Board/Youth':'Snowboard/Board/Youth',
'Snowboard/Boots/Mens':'Snowboard/Boots/Men',
'Snowboard/Boots/Womens':'Snowboard/Boots/Women',
'Snowboard/Boots/Youth':'Snowboard/Boots/Youth',
'Snowboard/Socks/Adult':'Snowboard/Accessory',
'Snowboard/Socks/Youth':'Snowboard/Accessory',
'Stupid/Misc/Crap':'Misc',
'Wakeboard/Accessory':'Watersport/Wakeboard/Accessory',
'Wakeboard/Bags':'Watersport/Wakeboard/Accessory',
'Wakeboard/Board/Mens':'Watersport/Wakeboard/Board',
'Wakeboard/Boots/Unisex':'Watersport/Wakeboard/Accessory',
'Wakeboard/Boots/Womens':'Watersport/Wakeboard/Accessory',
'Wakeboard/Packages/Unisex':'Watersport/Wakeboard/Package Deals',
'Wakeboard/Packages/Womens':'Watersport/Wakeboard/Package Deals',
'Wakeboard/Packages/Youth':'Watersport/Wakeboard/Package Deals',
'Wakeboard/Surf/Accessory':'Watersport/Wakeboard/Accessory',
'Wakeboard/Surf/Bag':'Watersport/Wakeboard/Accessory',
'Wakeboard/Wakeskate':'Watersport/Wakesurf',
'Wakeboard/Wakesurfs':'Watersport/Wakesurf',
'Watersport/Kneeboard/Board':'Watersport/Kneeboard',
'Watersport/Rashguard/Mens':'Watersport/Outfit/Rashguard',
'Watersport/Rashguard/Womens':'Watersport/Outfit/Rashguard',
'Watersport/SUP/Accessory':'Watersport/Paddleboard',
'Watersport/SUP/Hard':'Watersport/Paddleboard',
'Watersport/SUP/Inflatable':'Watersport/Paddleboard',
'Watersport/Ski/Accessory':'Watersport/Water Ski/Accessory',
'Watersport/Ski/Bag':'Watersport/Water Ski/Accessory',
'Watersport/Ski/Bindings':'Watersport/Water Ski/Accessory',
'Watersport/Ski/Combo':'Watersport/Water Ski/Combo',
'Watersport/Ski/Handle':'Watersport/Water Ski/Handle',
'Watersport/Ski/Single':'Watersport/Water Ski/Single',
'Watersport/Towable/Accessory':'Watersport/Towable/Accessory',
'Watersport/Towable/Tube':'Watersport/Towable/Tube',
'Watersport/Wetsuit/Mens':'Watersport/Outfit/Wetsuit',
'Watersport/Wetsuit/Womens':'Watersport/Outfit/Wetsuit',
'Watersport/Wetsuit/Youth':'Watersport/Outfit/Wetsuit',
'Winter/Equipment':'Misc',
'Women/Outerwear/Suit':'Gear/Outerwear',
'Womens/Baselayer/Bottom':'Gear/Base Layer/Women',
'Womens/Baselayer/Suit':'Gear/Base Layer/Women',
'Womens/Baselayer/Top':'Gear/Base Layer/Women',
'Womens/Lifestyle/Accessory':'Lifestyle/Women/Accessory',
'Womens/Lifestyle/Dress':'Lifestyle/Women/Dress',
'Womens/Lifestyle/Jacket':'Lifestyle/Women/Jacket',
'Womens/Lifestyle/Jumpsuit':'Lifestyle/Women',
'Womens/Lifestyle/Pants':'Lifestyle/Women/Pants',
'Womens/Lifestyle/Shoes':'Lifestyle/Women/Shoes',
'Womens/Lifestyle/Shorts':'Lifestyle/Women/Shorts',
'Womens/Lifestyle/Top':'Lifestyle/Women/Tops',
'Womens/Midlayer':'Gear/Midlayer/Women',
'Womens/Outerwear/Gloves':'Gear/Gloves/Adult',
'Womens/Outerwear/Jacket':'Gear/Jacket/Women',
'Womens/Outerwear/Mittens':'Gear/Mittens/Adult',
'Womens/Outerwear/Pants':'Gear/Pants/Women',
'Womens/Swimwear':'Lifestyle/Women/Swimwear',
'Youth/Baselayer/Bottom':'Gear/Base Layer/Youth',
'Youth/Baselayer/Suit':'Gear/Base Layer/Youth',
'Youth/Baselayer/Top':'Gear/Base Layer/Youth',
'Youth/Outerwear/Gloves':'Gear/Gloves/Youth',
'Youth/Outerwear/Jacket':'Gear/Jacket/Youth',
'Youth/Outerwear/Mittens':'Gear/Mittens/Youth',
'Youth/Outerwear/Pants':'Gear/Pants/Youth'}
#%% 
df = pd.read_pickle('df.pkl').reset_index(drop=False)  



df['webName'] = (df.name.str.title() + ' ' + df.year.fillna('')).str.strip()
df['itemType'] = np.nan
df.drop(columns = ['name','year','qty0','qty1','pSWAP','fCreated','fRcvd',
                    'lRcvd','lSold','lEdit','VC','DCS','D','C','S','isid'], inplace=True)
df.CAT = df.CAT.map(dCAT)

#%%

# DELETE ME ONCE CAT(egorie)S ARE GOOD
df.CAT = df.CAT.fillna('Misc')
df.BRAND = df.BRAND.str.title()

def options(row):
    if pd.notnull(row.color) and pd.notnull(row["size"]):
        row.webName = f'[RT]Color={row["color"]}'+','+f'[RB]Size={row["size"]}'
    elif pd.notnull(row["size"]) and pd.isnull(row["color"]):
        row.webName = f'[RB]Size={row["size"]}'
    elif pd.notnull(row["color"]) and pd.isnull(row["size"]):
        row.webName = f'[RT]Color={row["color"]}'
    return row



def convert(x):
    if len(x.index) > 1:
        fr = x.iloc[0:1,:].copy()
        # print(fr[cols])
        fr.color = np.nan
        fr['size'] = np.nan
        if x.description.notna().sum():
            fvi = x.description.first_valid_index()
            fv = x.loc[fvi,'description']
            fr.description =  fv
        if x.pic0.notnull().sum():
            fvi = x.pic0.first_valid_index()
            fv = x.loc[fvi,'pic0']
            fr.pic0 = fv
        
        fr.qty = x.qty.sum()
        fr.itemType = 'Product'
        fr.sku = '0-' + fr.sku
        fr.cost = x.cost.max()
        fr.pMAP = x.pSale.min()
        fr.pAmazon = x.pAmazon.min()
        
        """SKUs"""
        skus = x.copy()
        skus.itemType = 'SKU'
        skus.sku = '1-' + skus.sku
        
        skus = skus.apply(\
                    lambda x: options(x), axis = 1)
        
        """RULES"""
        rules = x.copy()
        rules.itemType = 'Rule'
        rules.pMAP = '[FIXED]' + rules.pSale.astype(str)
        # rules.pAmazon = '[FIXED]' + rules.pAmazon.astype(str)
        # rules.pSale = '[FIXED]' + rules.pSale.astype(str)
        # rules.pMAP = '[FIXED]' + rules.pMAP.astype(str)
        # rules.pMSRP = '[FIXED]' + rules.pMSRP.astype(str)
        rules.sku = '1-' + rules.sku
        
        return fr.append(skus).append(rules)
    else:
        x.sku = '2-' + x.sku
        x.itemType = 'Product'
        return x   

df = df.groupby('webName').apply(convert).reset_index(drop=True)

# mods
df['PP'] = np.where(df.qty>0,'Y','N')
df['PV'] = np.where((df.pic0.notnull())&(df.qty>0),'Y','N')
df['words'] = df[['webName','BRAND', 'color', 'size']] \
    .fillna('').apply(' '.join, axis=1).str.replace(' ',', ')

df.to_pickle('optionDf.pkl')



#%%

# uploady time baby

df = pd.read_pickle('optionDf.pkl')

out = pd.DataFrame(columns = ['Item Type',
                                'Product ID',
                                'Product Name',
                                'Product Type',
                                'Product Code/SKU',
                                'Bin Picking Number',
                                'Brand Name',
                                'Option Set',
                                'Option Set Align',
                                'Product Description',
                                'Price',
                                'Cost Price',
                                'Retail Price',
                                'Sale Price',
                                'Fixed Shipping Cost',
                                'Free Shipping',
                                'Product Warranty',
                                'Product Weight',
                                'Product Width',
                                'Product Height',
                                'Product Depth',
                                'Allow Purchases?',
                                'Product Visible?',
                                'Product Availability',
                                'Track Inventory',
                                'Current Stock Level',
                                'Low Stock Level',
                                'Category',
                                'Product Image ID - 1',
                                'Product Image File - 1',
                                'Product Image Description - 1',
                                'Product Image Is Thumbnail - 1',
                                'Product Image Sort - 1',
                                'Product Image ID - 2',
                                'Product Image File - 2',
                                'Product Image Description - 2',
                                'Product Image Is Thumbnail - 2',
                                'Product Image Sort - 2',
                                'Search Keywords',
                                'Page Title',
                                'Meta Keywords',
                                'Meta Description',
                                'MYOB Asset Acct',
                                'MYOB Income Acct',
                                'MYOB Expense Acct',
                                'Product Condition',
                                'Show Product Condition?',
                                'Event Date Required?',
                                'Event Date Name',
                                'Event Date Is Limited?',
                                'Event Date Start Date',
                                'Event Date End Date',
                                'Sort Order',
                                'Product Tax Class',
                                'Product UPC/EAN',
                                'Stop Processing Rules',
                                'Product URL',
                                'Redirect Old URL?',
                                'GPS Global Trade Item Number',
                                'GPS Manufacturer Part Number',
                                'GPS Gender',
                                'GPS Age Group',
                                'GPS Color',
                                'GPS Size',
                                'GPS Material',
                                'GPS Pattern',
                                'GPS Item Group ID',
                                'GPS Category',
                                'GPS Enabled'])


dCol = {'Cost Price':'cost',
        'Retail Price':'pMSRP',
        'Product Name':'webName',
        'Price':'pMAP',
        'Sale Price':'pSale',
        'Product Description':'description',
        'Product Warranty':'short_description',
        'Brand Name':'BRAND',
        'Current Stock Level':'qty',
        'Product Code/SKU':'sku',
        'Product UPC/EAN':'UPC',
        'Category':'CAT',
        'Option Set':'webName',
        'Item Type':'itemType',
        'Product Visible?':'PV',
        'Allow Purchases?':'PP',
        'Search Keywords':'words',
        'Meta Keywords':'words',
        'Meta Description':'description',
        'Page Title':'webName'}

for k,v in dCol.items():
    out[k] = df[v]

# del dCol, k, v

out[[f'Product Image File - {i+1}' for i in range(5)]]\
    = df[[f'pic{i}' for i in range(5)]]
out = out.reindex(out.columns.to_list()\
                  +[f"Product Image ID - {i}" for i in range(3,6)]\
                  +[f'Product Image Description - {i}' for i in range(3,6)],\
                      axis=1)
for i in range(1,6):
    out[f"Product Image Description - {i}"] = df.words
    out[f'Product Image Sort - {i}'] = i
out["Product Type"] = 'P'

# del i



# sparsify data.
out.loc[out['Item Type']=='Rule',\
        ['Product Name',\
         'Cost Price',\
         'Free Shipping',\
         'Current Stock Level',\
         'Low Stock Level',\
         'Product UPC/EAN']] = ''
out.loc[out['Item Type']=='SKU',\
        [c for c in out.columns \
         if c.startswith('Product Image')]] = ''        
out.loc[out['Item Type']=='SKU','Price'] = '' 
out.loc[out['Item Type'].isin(['SKU','Rule']),\
        ['Product Type',\
         'Brand Name',\
         'Product Description',\
         'Fixed Shipping Cost',\
         'Product Warranty',\
         'Product Weight',\
         'Search Keywords',\
         'Page Title',\
         'Meta Description',\
         'Product Condition',\
         'Show Product Condition?',\
         'Category',\
         'Meta Keywords',\
         'Meta Description',\
         'Option Set']] = ''
out.loc[out['Product Code/SKU'].str.contains('0-'),'Track Inventory']\
    = 'by option'    
out.loc[out['Product Code/SKU'].str.contains('2-'),'Track Inventory']\
    = 'by product'
out['Product Inventoried'] = 'Y'
out.replace('',np.nan, inplace=True)
out.dropna(axis=1, how='all', inplace=True)


out['Product Weight'] = 0


file =glob('in/*')[0]
ix = ['Item Type', 'Product Code/SKU']
# ix = ['itemType','sku']
imgCols = ['Product Image ID - 1',
 'Product Image File - 1',
 'Product Image Path - 1',
 'Product Image URL - 1',
 'Product Image Description - 1',
 'Product Image Is Thumbnail - 1',
 'Product Image Index - 1',
 'Product Image ID - 2',
 'Product Image File - 2',
 'Product Image Path - 2',
 'Product Image URL - 2',
 'Product Image Description - 2',
 'Product Image Is Thumbnail - 2',
 'Product Image Index - 2',
 'Product Image ID - 3',
 'Product Image File - 3',
 'Product Image Path - 3',
 'Product Image URL - 3',
 'Product Image Description - 3',
 'Product Image Is Thumbnail - 3',
 'Product Image Index - 3',
 'Product Image ID - 4',
 'Product Image File - 4',
 'Product Image Path - 4',
 'Product Image URL - 4',
 'Product Image Description - 4',
 'Product Image Is Thumbnail - 4',
 'Product Image Index - 4',
 'Product Image ID - 5',
 'Product Image File - 5',
 'Product Image Path - 5',
 'Product Image URL - 5',
 'Product Image Description - 5',
 'Product Image Is Thumbnail - 5',
 'Product Image Index - 5']

comp = pd.read_csv(file)\
    .rename(columns={\
                    'Product Visible':'Product Visible?',
                    'Allow Purchases':'Allow Purchases?',
                    'Description':'Product Description',
                    'Product SKU':'Product Code/SKU',
                    'Category String':'Category',
                    'Weight':'Product Weight'\
              })

comp.loc[:,ix[0]]=comp.loc[:,ix[0]].str.strip()
comp.set_index(ix,inplace=True)
# for production
end = dt.now()
os.rename(file,f'archive/ins/in_{str(end.date())} {end.hour}-{end.minute}-{end.second}.csv')
cols = imgCols\
    +['Product ID','Product Description']
out = out.reindex(list(set(out.columns.tolist()+cols)),axis=1).set_index(ix)
out.update(comp.loc[:,cols])
out = out.reset_index()
out['Product ID'] = out['Product ID'].fillna('-1.0').astype(float).astype(int).astype(str).replace('-1',np.nan)

#%%

out.to_csv('out/out.csv', quotechar="\"", index=False)

out.to_pickle(f'archive/outs/out_{str(end.date())} {end.hour}-{end.minute}-{end.second}.pkl')

print('Process complete!')
# del beginning, end, df, out
























