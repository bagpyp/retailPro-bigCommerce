# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:39:10 2020
@author: rtc@bagpyp.net
"""

# API CREDENTIALS

print('this process takes roughly ten minutes, please be patient.')

from datetime import datetime as dt
from datetime import timedelta as td
beginning = dt.now()
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
import requests
import json
import os
pd.options.display.max_columns = 80
pd.options.display.max_rows = 1000
pd.options.display.width = 150

#%% RUN ECM
print('beginning EM station procout ~3min')

os.system('T: && cd ECM && ecmproc -out -a -stid:001001A')

print('ECM procout complete, consuming XML files from RetailPro ~2min')

#%% PULL INVENTORY

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

df = pd.DataFrame(inventorys).replace('',np.nan).dropna(axis=1, how='all')
df = df[df.active=='1']
for col in df.columns:
    if df[col].nunique() == 1:
        df.drop(col,inplace=True,axis=1)
        


"""CODES"""

#vender
tree = ET.parse('T:/ECM/Polling/001001A/OUT/Vendor.xml')
root = tree.getroot()

vendors = []

vens = root.findall('./VENDORS/VENDOR')
for i in range(len(vens)):
    vendors.append(vens[i].attrib)

v = pd.DataFrame(vendors)
v = v[v.active=='1']
v = v.iloc[:,[0,2]].rename(columns={'vend_name':'BRAND'})

#cats
tree = ET.parse('T:/ECM/Polling/001001A/OUT/DCS.xml')
root = tree.getroot()

categories = []

cats = root.findall('./DCSS/DCS')
for i in range(len(cats)):
    categories.append(cats[i].attrib)

c = pd.DataFrame(categories)
c = c[c.active=='1'].iloc[:,[0,2,3,4]]
# format text
c.iloc[:,1:]=c.iloc[:,1:].applymap(str.title)

# dropping all DCS values that are not 6 or 9 characters long
c = c[(c.dcs_code.str.len()==6)|(c.dcs_code.str.len()==9)]\
    .reset_index(drop=True)

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

# putting cats and vends together
df = pd.merge(df, c, on='dcs_code', how='left', sort=False)
df = pd.merge(df, v, on='vend_code', how='left', sort=False)

#naming and ordering df
names = {'alu':'sku','local_upc':'UPC','description4':'alu','CAT':'CAT',
    'BRAND':'BRAND','description1':'name','description2':'year',
    'attr':'color','description3':'alt_color','siz':'size','qty_0':'qty0',
    'qty_1':'qty1','qty_250':'qty','cost':'cost','price_1':'pSale',
    'price_2':'pMAP','price_3':'pMSRP','price_4':'pAmazon',
    'price_5':'pSWAP',
    'created_date':'fCreated','fst_rcvd_date':'fRcvd',
    'lst_rcvd_date':'lRcvd','lst_sold_date':'lSold','modified_date':'lEdit',
    'vend_code':'VC','dcs_code':'DCS','D':'D',
    'C':'C','S':'S','style_sid':'ssid','item_sid':'isid','upc':'UPC2'}

df.rename(columns=names, inplace=True)
df = df[list(names.values())]


""" THIS IS WHERE THE PICKLE'S AT! """

# risky filter:
df = df[df.lEdit>str(dt.now()-td(days=5))]

df.to_pickle('fromECM.pkl')

#%% correcting dtypes, filtering further and tidying data
df = pd.read_pickle('fromECM.pkl')

print('Transforming data from RetailPro to BigCommerce-consumable format, ~1min')

#dtypes
for i in range(10,19): df.iloc[:,i] = pd.to_numeric(df.iloc[:,i])
df.iloc[:,13:19] = df.iloc[:,13:19].round(2)
df.iloc[:,10:13] = df.iloc[:,10:13].fillna(0).astype(int)
for i in range(19,24): df.iloc[:,i] = pd.to_datetime(df.iloc[:,i])
df.lEdit = df.lEdit.dt.tz_localize(None)


#filtering by category
surDf = df[~df.DCS.str.match(r'^((?!USD|REN|SER).)*$')]
surDf.to_csv('usedRentalAndService.csv')
df = df[df.DCS.str.match(r'^((?!USD|REN|SER).)*$')]

# filters products without UPCs w/ length 11, 12 or 13.
df = df[(df.UPC.str.len().isin([11,12,13]))]


df.drop('UPC2', axis=1, inplace=True)

#sort by sku
df.sku=df.sku.astype(int)
df = df.sort_values(by='sku')
df.sku = df.sku.astype(str).str.zfill(5)

df.reset_index(drop=True, inplace=True)
df.set_index('sku', inplace=True)

img = pd.read_csv('images.csv')
desc = pd.read_csv('descriptions.csv')
def f(df):
    df = df.sort_values(by='sku')
    df.sku = df.sku.astype(str).str.zfill(5)
    df.set_index('sku',inplace=True)
    return df
img = f(img)
desc = f(desc)

df= df.join(img, how='left').join(desc, how='left')
df.to_pickle('df.pkl')


#%% BEGIN BIGCOMMERCE TRANSFORMATION

df = pd.read_pickle('df.pkl').reset_index(drop=False)  

dCAT = {\
'Disc Golf/Bag':'Misc',
'Electronic/Audio':'Misc',
'Electronic/Camera/Accessory':'Misc',
'Eyewear/Goggles/Accessory':'Mountain Gear/Goggles/Accessory',
'Eyewear/Goggles/Moto':'Mountain Gear/Goggles/Accessory',
'Eyewear/Goggles/Rep. Lens':'Mountain Gear/Goggles/Accessory',
'Eyewear/Goggles/Unisex':'Mountain Gear/Goggles/Adult',
'Eyewear/Goggles/Womens':'Mountain Gear/Goggles/Women',
'Eyewear/Goggles/Youth':'Mountain Gear/Goggles/Youth',
'Eyewear/Sunglasses/Accessory':'Lifestyle/Sunglasses/Accessory',
'Eyewear/Sunglasses/Unisex':'Lifestyle/Sunglasses/Unisex',
'Eyewear/Sunglasses/Womens':'Lifestyle/Sunglasses/Womens',
'Headwear/Beanie':'Mountain Gear/Headwear/Beanie',
'Headwear/Facemask':'Mountain Gear/Headwear/Facemask',
'Headwear/Hat':'Lifestyle/Accessory',
'Hike/Pack/Accessory':'Lifestyle/Accessory',
'Hike/Pack/Hydration':'Lifestyle/Accessory',
'Hike/Pack/Map/Book':'Lifestyle/Accessory',
'Hike/Pack/Mens/Shoes':'Lifestyle/Men/Shoes',
'Hike/Pack/Womens/Shoes':'Lifestyle/Women/Shoes',
'Kayak/Accessory':'Watersport/Kayak',
'Lifejacket/Neoprene/Dog':'Watersport/Life Jackets/Dog',
'Lifejacket/Neoprene/Men':'Watersport/Life Jackets/Men',
'Lifejacket/Neoprene/Womens':'Watersport/Life Jackets/Women',
'Lifejacket/Neoprene/Youth':'Watersport/Life Jackets/Youth',
'Lifejacket/Nylon/Men':'Watersport/Life Jackets/Men',
'Lifejacket/Nylon/Womens':'Watersport/Life Jackets/Women',
'Lifejacket/Nylon/Youth':'Watersport/Life Jackets/Youth',
'Mens/Baselayer/Bottom':'Mountain Gear/Base Layer/Men',
'Mens/Baselayer/Suit':'Mountain Gear/Base Layer/Men',
'Mens/Baselayer/Top':'Mountain Gear/Base Layer/Men',
'Mens/Lifestyle/Accessory':'Lifestyle/Men/Accessory',
'Mens/Lifestyle/Bag':'Lifestyle/Men/Accessory',
'Mens/Lifestyle/Jacket':'Lifestyle/Men/Jacket',
'Mens/Lifestyle/Pants':'Lifestyle/Men/Pants',
'Mens/Lifestyle/Shoes':'Lifestyle/Men/Shoes',
'Mens/Lifestyle/Shorts':'Lifestyle/Men/Shorts',
'Mens/Lifestyle/Top':'Lifestyle/Men/Tops',
'Mens/Midlayer':'Mountain Gear/Midlayer/Men',
'Mens/Outerwear/Gloves':'Mountain Gear/Gloves/Men',
'Mens/Outerwear/Jackets':'Mountain Gear/Jacket/Men',
'Mens/Outerwear/Mittens':'Mountain Gear/Gloves/Women',
'Mens/Outerwear/Pants':'Mountain Gear/Pants/Men',
'Mens/Outerwear/Suit':'Mountain Gear/Outerwear',
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
'Ski/Bags/Mountain Gear/':'Ski/Accessory',
'Ski/Bags/Ski':'Ski/Accessory',
'Ski/Bindings/Mens':'Ski/Bindings/Men',
'Ski/Bindings/Womens':'Ski/Bindings/Women',
'Ski/Bindings/Youth':'Ski/Bindings/Youth',
'Ski/Boots/Accessory':'Ski/Accessory',
'Ski/Boots/Liner':'Ski/Accessory',
'Ski/Boots/Mens':'Ski/Boots/Men',
'Ski/Boots/Parts':'Ski/Boots/Parts',
'Ski/Boots/Womens':'Ski/Boots/Women',
'Ski/Boots/Youth':'Ski/Boots/Youth',
'Ski/Poles/Accessory':'Ski/Accessory',
'Ski/Poles/Adult':'Ski/Poles/Adult',
'Ski/Poles/Baskets':'Ski/Accessory',
'Ski/Poles/Youth':'Ski/Poles/Youth',
'Ski/Skis/Mens':'Ski/Skis/Men',
'Ski/Skis/Womens':'Ski/Skis/Women',
'Ski/Skis/Youth':'Ski/Skis/Youth',
'Ski/Socks/Adult':'Ski/Socks/Adult',
'Ski/Socks/Youth':'Ski/Socks/Youth',
'Ski/Tune/Wax':'Ski/Accessory',
'Ski/Tuning/Tool':'Ski/Accessory',
'Ski/X-Country/Bindings':'Ski/Accessory',
'Ski/X-Country/Boots':'Ski/Accessory',
'Ski/X-Country/Skis':'Ski/Accessory',
'Snowboard/Accessory':'Snowboard/Accessory',
'Snowboard/Bags/Backpack':'Snowboard/Bags/Backpack',
'Snowboard/Bags/Board Bag':'Snowboard/Bags/Board',
'Snowboard/Bags/Mountain Gear/':'Snowboard/Bags/Mountain Gear/',
'Snowboard/Bags/Travel':'Snowboard/Bags/Travel',
'Snowboard/Bags/Wheel':'Snowboard/Bags/Wheel',
'Snowboard/Bindings/Unisex':'Snowboard/Bindings/Adult',
'Snowboard/Bindings/Women':'Snowboard/Bindings/Women',
'Snowboard/Bindings/Youth':'Snowboard/Bindings/Youth',
'Snowboard/Board/Mens':'Snowboard/Board/Men',
'Snowboard/Board/Womens':'Snowboard/Board/Women',
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
'Women/Outerwear/Suit':'Mountain Gear/Outerwear',
'Womens/Baselayer/Bottom':'Mountain Gear/Base Layer/Women',
'Womens/Baselayer/Suit':'Mountain Gear/Base Layer/Women',
'Womens/Baselayer/Top':'Mountain Gear/Base Layer/Women',
'Womens/Lifestyle/Accessory':'Lifestyle/Women/Accessory',
'Womens/Lifestyle/Dress':'Lifestyle/Women/Dress',
'Womens/Lifestyle/Jacket':'Lifestyle/Women/Jacket',
'Womens/Lifestyle/Jumpsuit':'Lifestyle/Women',
'Womens/Lifestyle/Pants':'Lifestyle/Women/Pants',
'Womens/Lifestyle/Shoes':'Lifestyle/Women/Shoes',
'Womens/Lifestyle/Shorts':'Lifestyle/Women/Shorts',
'Womens/Lifestyle/Top':'Lifestyle/Women/Tops',
'Womens/Midlayer':'Mountain Gear/Midlayer/Women',
'Womens/Outerwear/Gloves':'Mountain Gear/Gloves/Women',
'Womens/Outerwear/Jacket':'Mountain Gear/Jacket/Women',
'Womens/Outerwear/Mittens':'Mountain Gear/Mittens/Women',
'Womens/Outerwear/Pants':'Mountain Gear/Pants/Women',
'Womens/Swimwear':'Lifestyle/Women/Swimwear',
'Youth/Baselayer/Bottom':'Mountain Gear/Base Layer/Youth',
'Youth/Baselayer/Suit':'Mountain Gear/Base Layer/Youth',
'Youth/Baselayer/Top':'Mountain Gear/Base Layer/Youth',
'Youth/Outerwear/Gloves':'Mountain Gear/Gloves/Youth',
'Youth/Outerwear/Jacket':'Mountain Gear/Jacket/Youth',
'Youth/Outerwear/Mittens':'Mountain Gear/Mittens/Youth',
'Youth/Outerwear/Pants':'Mountain Gear/Pants/Youth'}

df['webName'] = (df.name.str.title() + ' ' + df.year.fillna('')).str.strip()
df['itemType'] = np.nan
df.drop(columns = ['name','year','qty0','qty1','pSWAP','fCreated','fRcvd',
                    'lRcvd','lSold','lEdit','VC','DCS','D','C','S','isid'], inplace=True)

# map to web categories
df.CAT = df.CAT.map(dCAT)

# DELETE ME ONCE CAT(egorie)S ARE GOOD
df.CAT = df.CAT.fillna('Misc')

#title out brands
df.BRAND = df.BRAND.str.title()

#words
words = df[['webName','BRAND', 'color', 'size','alu','sku','description']]
df['words'] = words.fillna('').apply(' '.join,axis=1).str.lower()
df['words_short'] = words.drop('description',axis=1)\
    .fillna('').apply(' '.join,axis=1).str.lower()

"""COLORS"""
# made everything Terracotta And Black, use with caution
# cMap = df[['color','alt_color']].dropna()
# cMap = cMap[~cMap.alt_color.str.contains(r'[0-9]')]
# cMap = cMap.drop_duplicates()
# cMap = cMap[cMap.color!=cMap.alt_color]
# cMap = cMap[~(cMap.alt_color.str.contains('COMPLETE')\
#             | cMap.alt_color.str.contains('TRIM-TO-FIT'))]
# for s in [' /','/ ',' / ']:
#     cMap.alt_color = cMap.alt_color.replace(s,'/')
# cMap.alt_color = cMap.alt_color.str.title()
# cMap = cMap.set_index('color',drop=True)
# cMap.to_csv('colorMap.csv')
cMap = pd.read_csv('colorMap.csv').set_index('color',drop=True)
cMap = cMap.to_dict()

df.color = df.color.map(cMap['alt_color'])
df.drop('alt_color',axis=1,inplace=True)
"""
In [236]: cMap.nunique()
Out[236]: 
color        1900
alt_color    1724
dtype: int64
"""

df.to_pickle('preOp.pkl')

#%% CONFIGURE PRODUCT OPTIONS, GEROUPS BY webName

df = pd.read_pickle('preOp.pkl')

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
        
        
        """BASE PRODUCT"""
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
        fr.pSale = x.pSale.min()
        fr.pMSRP = x.pMSRP.max()
        fr.pMAP = x.pMAP.max()

        
        """SKUs"""
        skus = x.copy()
        skus.itemType = 'SKU'
        skus.sku = '1-' + skus.sku
        skus.cost = np.where(skus.cost!=skus.loc[skus.index.min(),'cost'],\
                             '[FIXED]' + skus.cost.astype(str),\
                             np.nan)
        skus = skus.apply(\
                    lambda x: options(x), axis = 1)
        
        """RULES"""
        rules = x.copy()
        rules.itemType = 'Rule'
        for col in ['pSale','pMAP','pMSRP']:
            rules[col] = np.where(rules[col]!=rules.loc[rules.index.min(),col],\
                                  '[FIXED]' + rules[col].astype(str),\
                                  np.nan)
        rules.sku = '1-' + rules.sku
        
        return fr.append(skus).append(rules)
    else:
        x.sku = '2-' + x.sku
        x.itemType = 'Product'
        return x   

print('configuring product options')

df = df.groupby('webName',sort=False).apply(convert).reset_index(drop=True)
df = df.replace('[FIXED]nan',np.nan)
df.to_pickle('optionDf.pkl')

print('options configured, issuing API call to BigCommerce to pull changes from admin panel.')

#%% API CALL TO BC

def productIDs(n):
    # &include_fields=sku
    path = base + f'v3/catalog/products?limit=250&page={n}&include=variants,images'
    res = requests.get(url=path, headers=headers)
    return json.loads(res.text)

firstPage = productIDs(1)
data = firstPage['data']
totalPages = firstPage['meta']['pagination']['total_pages']
for i in range(2,totalPages+1):
    #currently 31 pages
    data.extend(productIDs(i)['data'])
    
print('API call complete')

pIds = pd.json_normalize(data)[['id','sku','variants','images']]
vIds = pd.json_normalize(pIds.variants.sum())[['sku_id','sku']]\
    .rename({'sku_id':'id'}, axis=1).dropna()
pIds.drop('variants',axis=1,inplace=True)
vIds.id = vIds.id.astype(int).astype(str)
pIds['Item Type'] = 'Product'
vIds['Item Type'] = 'SKU'
pIds['Product Image File - 1'] = pIds.images.astype(str)\
                                        .replace('[]',np.nan)
pIds.drop('images',axis=1,inplace=True)
ids = pd.concat([pIds,vIds]).astype(str)
ids = ids.rename({'sku':'Product Code/SKU',
                  'id':'Product ID'},axis=1)\
        .set_index('Product Code/SKU')


#%% CONFIGURE THE OUT FILE
# sparsifies data near end of cell

df = pd.read_pickle('optionDf.pkl')

out = pd.DataFrame(columns = ['Item Type',
                                'Product Name',
                                'Product Code/SKU',
                                'Brand Name',
                                'Option Set',
                                'Product Description',
                                'Price',
                                'Cost Price',
                                'Retail Price',
                                'Sale Price',
                                'Product Warranty',
                                'Product Weight',
                                'Product Visible?',
                                'Current Stock Level',
                                'Track Inventory',
                                'Category',
                                'Search Keywords',
                                'Page Title',
                                'Meta Keywords',
                                'Meta Description',
                                'Product UPC/EAN'])

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
        'Search Keywords':'words_short',
        'Meta Keywords':'words',
        'Meta Description':'description',
        'Page Title':'webName'}

for k,v in dCol.items():
    out[k] = df[v]

out[[f'Product Image File - {i+1}' for i in range(5)]]\
    = df[[f'pic{i}' for i in range(5)]]
for i in range(1,6):
    out.loc[out[f'Product Image File - {i}'].notna(),\
            f'Product Image Description - {i}']=df.words_short
    
out["Product Type"] = 'P'
out['Product Weight'] = 0
out['Product Inventoried'] = 'Y'

# sparsify data
out.loc[out['Item Type']=='Rule',\
        ['Product Name',\
         'Cost Price',\
         'Current Stock Level',\
         'Product UPC/EAN']] = ''
        
out.loc[out['Item Type']=='SKU',['Price','Sale Price','Retail Price']\
                                + [c for c in out.columns \
                                   if c.startswith('Product Image')]] = '' 

out.loc[out['Item Type'].isin(['SKU','Rule']),\
        ['Product Type',\
         'Brand Name',\
         'Product Description',\
         'Product Warranty',\
         'Product Weight',\
         'Search Keywords',\
         'Page Title',\
         'Meta Description',\
         'Category',\
         'Meta Keywords',\
         'Meta Description',\
         'Option Set']] = ''
    
out.loc[out['Product Code/SKU'].str.contains('0-'),'Track Inventory']\
    = 'by option'    
out.loc[out['Product Code/SKU'].str.contains('2-'),'Track Inventory']\
    = 'by product'

out.replace('',np.nan, inplace=True)

        
out['Product ID'] = np.nan
out.set_index('Product Code/SKU',inplace=True)
out.update(ids[['Product ID','Product Image File - 1']])
out.reset_index(inplace=True)

# sparsify (almost) all data for products that only need to be updated
cols = [\
  # 'Item Type',
  # 'Product Code/SKU',
  # 'Product ID',
  # 'Product Name',
  'Brand Name',
  'Option Set',
  'Product Description',
  # 'Price',
  # 'Cost Price',
  # 'Retail Price',
  # 'Sale Price',
  # 'Product Warranty',
  'Product Weight',
  'Product Visible?',
  'Track Inventory',
  # 'Current Stock Level',
  'Category',
  'Search Keywords',
  'Page Title',
  'Meta Keywords',
  'Meta Description',
  'Product UPC/EAN',
  # 'Product Image File - 1',
  'Product Image File - 2',
  'Product Image File - 3',
  'Product Image File - 4',
  'Product Image File - 5',
  'Product Image Description - 1',
  'Product Image Description - 2',
  'Product Image Description - 3',
  'Product Image Description - 4',
  'Product Image Description - 5',
  'Product Type',
  'Product Inventoried']
    
out.loc[out['Product ID'].notna(),cols]=np.nan

out.loc[out['Item Type']=='Rule','Product ID'] = np.nan
out['Product Image File - 1'] = out['Product Image File - 1'].replace('nan',np.nan)
out.loc[out['Item Type']=='Product','Product Visible?'] \
    = np.where(\
        out.loc[out['Item Type']=='Product','Product Image File - 1'].notna()\
        ,'Y','N')

out.loc[:,'Product Image File - 1'] = np.nan

#%% WRITE TO CSV 
end = dt.now()
out = out.dropna(thresh=4)
out.dropna(how='all',axis=1,inplace=True)
out.reset_index(inplace=True, drop=True)
out['Sort Order'] = np.where(out['Item Type']=='Product',\
                             [len(out)-i for i in out.index],\
                             np.nan)
out = out[out.columns[-2:].tolist()+out.columns[:-2].tolist()]
out.to_csv('out/out.csv', quotechar="\"", index=False)
# out.to_pickle(f'archive/outs/out_{str(end.date())} {end.hour}-{end.minute}-{end.second}.pkl')

#%% ORDERS

def getAllOrders():
    path = base + 'v2/orders'
    # ?min_date_created=<string>
    # <string>:
    """ 
    Minimum date the order was created in RFC-2822 or ISO-8601.
    RFC-2822: `Thu, 20 Apr 2017 11:32:00 -0400`
    ISO-8601: `2017-04-20T11:32:00.000-04:00`
    """
    res = requests.get(url=path, headers=headers)
    return json.loads(res.text)

def getOrderProducts(i):
    path = base + f'v2/orders/{i}/products'
    res = requests.get(url=path, headers=headers)
    return json.loads(res.text)

def getOrderShipping(i):
    path = base + f'v2/orders/{i}/shipping_addresses'
    res = requests.get(url=path, headers=headers)
    return json.loads(res.text)


orders = getAllOrders()
oDf = pd.json_normalize(orders)



names = [os.path.basename(x) for x in glob('orders/*')]
for i in oDf.id.tolist():
    o = oDf[oDf.id==i].reset_index()
    p = pd.json_normalize(getOrderProducts(i))
    p.columns = 'p.' + p.columns
    s = pd.json_normalize(getOrderShipping(i))
    s.columns = 's.' + s.columns
    if str(i) + '.txt' not in names:
        with open(f'orders/{i}.txt','w') as file:
            file.write(o.status.iloc[0] + '\n\n')
            file.write(str(pd.concat([o,p,s],axis=1).T))
            
end = dt.now()
print(f'Process completed in ~{int(((end-beginning).seconds)/60)} minutes\n\n',
      'Please upload "out.csv" to BigCommerce at your nearest convenience.\n',
      'from hillcrestsports.com/admin, select Products > Import\n',
      'choose file "out.csv"\n',
      'Select "Overwrite Existing Products,\n\t"Next>"\n\t"Start Import"')
