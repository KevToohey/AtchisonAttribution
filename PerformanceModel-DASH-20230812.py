#!/usr/bin/env python
# coding: utf-8

# ![convert notebook to web app](https://raw.githubusercontent.com/KevToohey/AtchisonPublic/main/logo.png)

# 
# <html>
# <body style="background-color:#3D555E;">
# 
#    
# <h1 style="color:#93F205;">Portfolio Reporting and Attribution Model</h1>
# 
# <p style="color:#E7EAEB;">This model attempts to provide a scalable reporting model to provide daily portfolio monitoring, performance and attribution analysis, and client communications
# </p>
# <p style="color:#E7EAEB;">Last Edits: Kev Toohey - 10 August 2023
# </p><br>
# <br>
# 

# In[1]:


# ----------- COMMAND LINE Functions -------------

#to hide code in HTML output via terminal:
#->  jupyter nbconvert PerformanceModel-20230809.ipynb --to html --no-input 

#to create a steam dashboard run via terminal:

# jupyter nbconvert --to python PerformanceModel-20230723.ipynb
# streamlit run PerformanceModel-20230723.py
# You can then access the dashboard by going to link localhost:8501

# ----------- ---------------------- -------------

#Overview of Method 
#STAGE 1: Import Asset Framework
#Firstly, a set of asset lines are imported. Assets are categorised and daily return series are imported via a Github registry csv files

#A set of portfolio weights csv files are then able to be imported.

#Initially 3x level of portfolio allocation will be attributed.

#1. A simple reference allocation portfolio
#2. Medium term target asset allocation = SAA
#3. Actual asset allocation = TAA

#Test data .csv files have been loaded onto a public GitHub registery:
#https://github.com/KevToohey/AtchisonAttribution/blob/main/WeightFile.csv
#https://github.com/KevToohey/AtchisonAttribution/blob/main/ProductList.csv
#https://github.com/KevToohey/AtchisonAttribution/blob/main/ProductDailyReturns.csv
#https://github.com/KevToohey/AtchisonAttribution/blob/main/BMGroup1.csv

# importing Libraries

#colour1 = "#3D555E"  #BG Grey/Green
#colour2 = "#E7EAEB"  #Off White
#colour3 = "#93F205"  #Green
#colour4 = "#1DC8F2"  #Blue
#colour5 = "#F27D11"  #Orange

import gc  #garbage collection
import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
plt.style.use('default')
#%matplotlib inline

import seaborn as sns

import datetime as dt
#import calendar
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
#import ipywidgets as widgets
#from ipywidgets import interact, interact_manual

#import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objects as go
#from  plotly.offline import plot
#import cufflinks as cf
#cf.go_offline()
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
#init_notebook_mode(connected='true')

import plotly.express as px  # (version 4.7.0 or higher)
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)

app = Dash(__name__)
server = app.server

#from IPython.display import display, clear_output
#from IPython.display import HTML

pd.options.display.max_columns = 40
pd.options.display.max_rows = 40


# In[2]:


# IMPORT Datafiles stored on Kev's GitHUB Registry

folderPath = 'https://raw.githubusercontent.com/KevToohey/AtchisonAttribution/main/'
#folderPath = 'C:/Users/kevto/Downloads/PythonModel/dataIn/'

#df_productList = pd.read_csv('https://raw.githubusercontent.com/KevToohey/AtchisonAttribution/main/ProductList.csv', index_col='Code')
df_productList = pd.read_csv(folderPath+'ProductList.csv', index_col='Code')

#df_BM_G1 = pd.read_csv('https://raw.githubusercontent.com/KevToohey/AtchisonAttribution/main/BMGroup1.csv')
df_BM_G1 = pd.read_csv(folderPath+'BMGroup1.csv')

#df_rawProductReturns_d = pd.read_csv('https://raw.githubusercontent.com/KevToohey/AtchisonAttribution/main/ProductDailyReturns.csv', index_col='Date', parse_dates=True)
df_rawProductReturns_d = pd.read_csv(folderPath+'ProductDailyReturns.csv', index_col='Date', parse_dates=True, dayfirst=True)

#df_rawWeightFile = pd.read_csv('https://raw.githubusercontent.com/KevToohey/AtchisonAttribution/main/WeightFile.csv', parse_dates=True)
df_rawWeightFile = pd.read_csv(folderPath+'WeightFile.csv', parse_dates=True, dayfirst=True)
df_rawWeightFile['Date'] = pd.to_datetime(df_rawWeightFile['Date'])

#df_rawLimitFile = pd.read_csv('https://raw.githubusercontent.com/KevToohey/AtchisonAttribution/main/PortfolioPolicyLimits.csv', parse_dates=True)
df_rawLimitFile = pd.read_csv(folderPath+'PortfolioPolicyLimits.csv', parse_dates=True, dayfirst=True)
df_rawLimitFile['Date'] = pd.to_datetime(df_rawLimitFile['Date'])

# remove blanks from column headings
df_productList.columns = df_productList.columns.str.replace(' ','')
df_BM_G1.columns = df_BM_G1.columns.str.replace(' ','')
df_rawProductReturns_d.columns = df_rawProductReturns_d.columns.str.replace(' ','')
df_rawWeightFile.columns = df_rawWeightFile.columns.str.replace(' ','')



# In[3]:


# Clean Data Input 1

# Remove blank rows, remove Position Rows where not included in Position, create row Sum

# Creates Error Term Value: err_missingProductCodes
# Creates Global Value:  Cat1List, Cat2List, validCodes
#ProductList = df_productList.index.unique()

## To do - check if G1List matches the unique list from df_BMGroup1 dataframe


df_rawWeightFile["RebalanceID"] = df_rawWeightFile["PortfolioCode"].astype(str) + df_rawWeightFile["AllocationType"].astype(str) + df_rawWeightFile["Date"].astype(str)
df_rawWeightFile = df_rawWeightFile[df_rawWeightFile.RebalanceID != 'nannanNaT']
rawSets = df_rawWeightFile.groupby(['RebalanceID'])['Weight'].sum()

df_rawLimitFile["LimitID"] = df_rawLimitFile["PortfolioCode"].astype(str) + df_rawLimitFile["Group"].astype(str) + df_rawLimitFile["GroupValue"].astype(str)
df_rawLimitFile = df_rawLimitFile[df_rawLimitFile.LimitID != 'nanNaTnannan']
# net to....check if min is < max

validCodes = df_productList.index.unique()
err_missingProductCodes = df_rawWeightFile[~df_rawWeightFile['Position'].isin(validCodes)]

df_cleanWeightFile = df_rawWeightFile
df_cleanWeightFile = df_rawWeightFile[df_rawWeightFile['Position'].isin(validCodes)]



# In[4]:


# Clean Data Input 2

# Remove rows where Sum weight != 100

# Creates Error Term Value:  err_invalidSet
# Creates Global Value:  validSets
# Creates Final cleans Dataframe Output:  df_validWeightFile

validSets = df_cleanWeightFile.groupby(['RebalanceID'])['Weight'].sum()
validSets = validSets[validSets == 100.0].index.tolist()
df_validWeightFile = df_cleanWeightFile[df_cleanWeightFile['RebalanceID'].isin(validSets)]
err_invalidSet = rawSets[~rawSets.index.isin(validSets)].index



# In[5]:


# Clean Data Input 3

# Check daily return data structure is suitable, remove Null Values, match and order column headings. 
# Only continue with return analysis if all columns present in Product List match Column Headings 
# from Returns input table, also fill zeros for weekends and holidays

# Creates Error Term Value:  err_CodesMatch that is False puts an end to analysis progressing
# Creates Final clean and ordered Return Output:  df_orderedProductReturns_d

df_cleanProductReturns_d = df_rawProductReturns_d.fillna(0)

if all(df_cleanProductReturns_d.columns.isin(df_productList.index)):
    if all(df_productList.index.isin(df_cleanProductReturns_d.columns)):
        err_CodesMatch = True
        df_orderedProductReturns_d = df_cleanProductReturns_d.reindex(columns=df_productList.index)
    else:
        err_CodesMatch = False
else:
    err_CodesMatch = False



# In[6]:


# CORE FUNCTIONS 1: Populating Daily Portfolio Weight Functions

def f_locateWeightsRow(df_input, validDates, dateLookup, portfolioCode, frameType):
    selectedList = [x for x in validDates if x.startswith(portfolioCode+frameType)]
    dateLookupString = dateLookup.strftime("%Y-%m-%d")
    selectedListDate = [x for x in selectedList if x.endswith(dateLookupString)]
    df_selectedDayWeight = df_input[df_input['RebalanceID'].isin(selectedListDate)]
    df_output = df_selectedDayWeight[['Position', 'Weight']].copy().set_index('Position')
    return df_output

def f_dateRange(startDate, endDate):
    for n in range(int((endDate - startDate).days+1)):
        yield startDate + dt.timedelta(n)
        
def f_portfolioWeightFrame(df_input, validDates, portfolioCode, df_input_BMGroup, frameType, startDate, endDate):
    df_output = pd.DataFrame(columns=df_productList.index.unique())
    groupList = df_input_BMGroup[df_input_BMGroup.columns[0]].unique()
    groupName = df_input_BMGroup.columns[0]
    
    for single_date in f_dateRange(startDate, endDate):
        single_date_string = single_date.strftime("%Y-%m-%d")

        if (portfolioCode+frameType+single_date_string) in validDates:
            df_output.loc[single_date] = 0
            df_newDayWeight = f_locateWeightsRow(df_input, validDates, single_date, portfolioCode, frameType)
            for n in range(len(df_newDayWeight.index)):
                df_output.loc[[single_date],df_newDayWeight.iloc[[n]].index] = float(df_newDayWeight.iloc[[n]].Weight)

        else:
            if single_date == startDate:
                df_output.loc[single_date] = 0
            else:
                df_output.loc[single_date] = df_output.loc[single_date - dt.timedelta(1)]

    df_output['P_TOTAL'] = df_output[df_productList.index.unique()].sum(axis=1, numeric_only=True)
    df_output['BM_TOTAL'] = df_output['P_TOTAL']
    
    #NEED TO remove hard coded G1 and look up column number to check if G1 or G2 or G3?
    for n in range(len(groupList)):
        groupList_n = df_productList.index[df_productList[groupName] == groupList[n]].tolist()
        df_output['P_'+groupName+'_'+ groupList[n]] = df_output[df_output.columns.intersection(groupList_n)].sum(axis=1, numeric_only=True)
        df_output['BM_'+groupName+'_'+ groupList[n]] = df_output['P_'+groupName+'_'+ groupList[n]]
        
    return df_output

def f_portfolioWeightSets(df_input, validDates, portfolioCode, df_Group, startDate, endDate):
    df_outputL1 = f_portfolioWeightFrame(df_input, validDates, portfolioCode, df_Group, 'Reference', startDate, endDate)
    df_outputL2 = f_portfolioWeightFrame(df_input, validDates, portfolioCode, df_Group, 'Strategic', startDate, endDate)
    df_outputL3 = f_portfolioWeightFrame(df_input, validDates, portfolioCode, df_Group, 'Tactical', startDate, endDate) 
    df_outputPeer = f_portfolioWeightFrame(df_input, validDates, portfolioCode, df_Group, 'Peer', startDate, endDate)   
    df_outputObj = f_portfolioWeightFrame(df_input, validDates, portfolioCode, df_Group, 'Objective', startDate, endDate) 
       
    return df_outputL1, df_outputL2, df_outputL3, df_outputPeer, df_outputObj



# In[7]:


# CORE FUNCTIONS 2: Populating Portfolio Return Framework & Contribution Framework

def f_addPortfolioReturnGroup(df_input_r, df_input_w, df_input_BMGroup, df_input_Products, startDate, endDate):
    
    # Create list of Groups names within defined Group Set
    groupList = df_input_BMGroup[df_input_BMGroup.columns[0]].unique()
    groupName = df_input_BMGroup.columns[0]
    productList = df_input_Products.index.unique()
    # Create Portfolio Returns
    df_output_r = df_input_r.loc[startDate:endDate]

    # Create Portfolio Contributions
    df_output_contrib = df_output_r.loc[:,df_output_r.columns.intersection(productList)] * (df_input_w.loc[:,df_input_w.columns.intersection(productList)] / 100)
    df_output_contrib['P_TOTAL'] = df_output_contrib.sum(axis=1, numeric_only=True) 
    df_output_contrib['BM_'+groupName+'_TOTAL'] = 0 #initialise as zero, put in table order, set value below by accretion
    
    for n in range(len(groupList)):
        GList_n = df_input_Products.index[df_input_Products[groupName] == groupList[n]].tolist()
        df_output_contrib['P_'+groupName+'_'+ groupList[n]] = df_output_contrib[df_output_contrib.columns.intersection(GList_n)].sum(axis=1, numeric_only=True)
        df_output_contrib['BM_'+groupName+'_'+ groupList[n]] = 0 #initialise as zero, put in table order, set value below in for loop
        
    # Create Portfolio Returns Sub-components by Daily Weighting Contributions
    df_output_r['P_TOTAL'] = df_output_contrib['P_TOTAL'] 
    df_output_r['BM_'+groupName+'_TOTAL'] = 0 ##initialise as zero, put in table order, set value below by accretion
    
    for n in range(len(groupList)):
        df_output_r['P_'+groupName+'_'+ groupList[n]] = df_output_contrib['P_'+groupName+'_'+groupList[n]] / (df_input_w['P_'+df_input_BMGroup.columns[0]+'_'+groupList[n]] / 100)    

        GBM_n = df_input_BMGroup.Code[df_input_BMGroup[groupName] == groupList[n]].tolist()
        df_output_r['BM_'+groupName+'_'+ groupList[n]] = df_output_r[GBM_n]
                
        df_output_contrib['BM_'+groupName+'_'+ groupList[n]] = (df_output_r['BM_'+groupName+'_'+ groupList[n]]) * (df_input_w['BM_'+df_input_BMGroup.columns[0]+'_'+groupList[n]] / 100) 
        # Loop accrete Total column values 
        df_output_contrib['BM_'+groupName+'_TOTAL'] += df_output_contrib['BM_'+groupName+'_'+ groupList[n]]
    
    df_output_r['BM_'+groupName+'_TOTAL'] = df_output_contrib['BM_'+groupName+'_TOTAL']
    
    return df_output_r, df_output_contrib
        


# In[8]:


# CORE FUNCTIONS 3: Populating Portfolio Attribution Framework

def f_2FactorBrinsonFachlerFrame(df_input_P_r, df_input_P_w, df_input_BM_r, df_input_BM_w, df_input_BMGroup, df_input_Products):
    # Method Used: Modified 2 Factor Brinson Fachler version assuming Top Down decision priority, 
    #        then bottom up decision - therefore interaction is calculated as part of selection 
    #        Theoretic process applied from CFA Chapter 5.

    groupList = df_input_BMGroup[df_input_BMGroup.columns[0]].unique()
    groupName = df_input_BMGroup.columns[0]
    productList = df_input_Products.index.unique()
    df_output_a = pd.DataFrame()
    
    df_output_a['P_TOTAL_'+groupName+' -- Allocation Effect'] = df_input_P_r['P_TOTAL']
    df_output_a['P_TOTAL_'+groupName+' -- Allocation Effect'] = 0
    df_output_a['P_TOTAL_'+groupName+' -- Selection Effect'] = 0            
    
    # Create Portfolio Contributions
    for n in range(len(groupList)):
        #Allocation Effect = (wi - Wi)(Bi-B)
        df_output_a[groupName+'_'+groupList[n]+'-- Allocation Effect'] = (((df_input_P_w['P_'+groupName+'_'+ groupList[n]]/100) - (df_input_BM_w['BM_'+groupName+'_'+ groupList[n]]/100)) * ((df_input_BM_r['BM_'+groupName+'_'+ groupList[n]]) - (df_input_BM_r['BM_'+groupName+'_TOTAL'])))
        df_output_a[groupName+'_'+groupList[n]+'-- Allocation Effect'] = df_output_a[groupName+'_'+groupList[n]+'-- Allocation Effect'].fillna(0)
        
        #Selection Effect = wi(Ri - Bi)
        df_output_a[groupName+'_'+groupList[n]+'-- Selection Effect'] = (df_input_P_w['P_'+groupName+'_'+groupList[n]]/100) * ((df_input_P_r['P_'+groupName+'_'+groupList[n]]) - (df_input_BM_r['BM_'+groupName+'_'+groupList[n]]))
        df_output_a[groupName+'_'+groupList[n]+'-- Selection Effect'] = df_output_a[groupName+'_'+groupList[n]+'-- Selection Effect'].fillna(0)
        
        # Accrete Total Portfolio Attribution Value        
        df_output_a['P_TOTAL_'+groupName+' -- Allocation Effect'] += df_output_a[groupName+'_'+groupList[n]+'-- Allocation Effect']
        df_output_a['P_TOTAL_'+groupName+' -- Selection Effect'] += df_output_a[groupName+'_'+groupList[n]+'-- Selection Effect']


    test1 = pd.DataFrame()
    test1['P_TOTAL'] = df_input_P_r['P_TOTAL']
    test1['IntEq_wi'] = (df_input_P_w['P_'+groupName+'_'+ groupList[1]]/100)
    test1['IntEq_Wi'] = (df_input_BM_w['BM_'+groupName+'_'+ groupList[1]]/100)
    test1['IntEq_Bi'] = (df_input_BM_r['BM_'+groupName+'_'+ groupList[1]])  
    test1['IntEq_B'] = (df_input_BM_r['BM_'+groupName+'_TOTAL'])
    
    test2 = pd.DataFrame()
    test2['P_TOTAL'] = df_input_P_r['P_TOTAL']
    test2['IntEq_wi'] = (df_input_P_w['P_'+groupName+'_'+groupList[1]]/100)
    test2['IntEq_Ri'] = (df_input_P_r['P_'+groupName+'_'+groupList[1]])
    test2['IntEq_Bi'] = (df_input_BM_r['BM_'+groupName+'_'+groupList[1]])
    
    return df_output_a, test1, test2

def f_SetLimitFrame(df_input_limits, df_input_BMGroup, portfolioCode):
    
    groupList = df_input_BMGroup[df_input_BMGroup.columns[0]].unique()
    groupName = df_input_BMGroup.columns[0]
    
    for n in range(len(df_input_BMGroup.index)):
        df_rawLimitFile[df_input_limits.LimitID == (portfolioCode+groupName+groupList[n])]
    
    return df_Output


# In[9]:


# CORE FUNCTIONS 4 - Calculation Return and Volatility Results

#Calculation Performance Index
def f_CalcReturnValues(df_Input, startDate, endDate):
# example use:  returnOutput = f_CalcReturnValues(df_L3_r, dates1.loc[1,'Date'], dates1.loc[0,'Date'])

    returnOutput = 0.0
    days = 0.0
    days = ((endDate - startDate).days)
    
    if days > 0: returnOutput = ((df_Input.loc[startDate+relativedelta(days= 1):endDate] + 1).cumprod() - 1).iloc[-1]
    elif days == 0: returnOutput = df_Input.iloc[0]
    else: returnOutput = 0 #throw error here

    if days > 365: returnOutput = (((1+ returnOutput) ** (1 / (days / 365)))-1)

    return returnOutput

def f_CalcReturnTable(df_Input, dateList):
# example use:  f_CalcReturnTable(df_L3_r.loc[:,['IOZ','IVV']], tME_dates)    

    df_Output = pd.DataFrame()
    
    for n in range(len(dateList)):
        if n > 0: df_Output[dateList.loc[n,'Name']] = f_CalcReturnValues(df_Input, dateList.loc[n,'Date'], dateList.loc[0,'Date'])
        
    return df_Output


# In[10]:


# CORE FUNCTIONS 5 - Set Dates

def f_SetDates(t_StartDate, t_EndDate):

    t_SOMDate = t_EndDate+relativedelta(days= -t_EndDate.day+1)
    t_CYDate = dt.date((t_EndDate.year-1), 12, 31)
 
    if t_EndDate.month < 7:
        t_FYDate = dt.date((t_EndDate.year-1), 6, 30)
    else:
        t_FYDate = dt.date((t_EndDate.year), 6, 30)

    # Define Date of last full month performance, and last full quarter performance
    if calendar.monthrange(t_EndDate.year, t_EndDate.month)[1] == t_EndDate.day:
        tME_EndDate = t_EndDate
    else:
        tME_EndDate = t_EndDate+relativedelta(days= -t_EndDate.day)

    tME_SOMDate = tME_EndDate+relativedelta(days= -tME_EndDate.day+1)

    # Find last Full quarter
    if tME_EndDate.month % 3 > 0:
        tQE_SOMDate = tME_EndDate+relativedelta(days= -tME_EndDate.day+1)+relativedelta(months= -(tME_EndDate.month % 3) )
        tQE_EndDate = tQE_SOMDate+relativedelta(days= (calendar.monthrange(tQE_SOMDate.year, tQE_SOMDate.month)[1]) -1)
    else:
        tQE_EndDate = tME_EndDate
        tQE_SOMDate = tME_SOMDate
    
    # Create lookback dates
    tME_1mDate = tME_SOMDate+relativedelta(days= -1)
    tME_2mDate = tME_SOMDate+relativedelta(months= -1)+relativedelta(days= -1)
    tME_3mDate = tME_SOMDate+relativedelta(months= -2)+relativedelta(days= -1)
    tME_6mDate = tME_SOMDate+relativedelta(months= -5)+relativedelta(days= -1)
    tME_12mDate = tME_SOMDate+relativedelta(months= -11)+relativedelta(days= -1)
    tME_24mDate = tME_SOMDate+relativedelta(months= -23)+relativedelta(days= -1) 
    tME_36mDate = tME_SOMDate+relativedelta(months= -35)+relativedelta(days= -1)
    tME_48mDate = tME_SOMDate+relativedelta(months= -47)+relativedelta(days= -1) 
    tME_60mDate = tME_SOMDate+relativedelta(months= -59)+relativedelta(days= -1) 
    tME_72mDate = tME_SOMDate+relativedelta(months= -71)+relativedelta(days= -1) 
    tME_84mDate = tME_SOMDate+relativedelta(months= -83)+relativedelta(days= -1)
    tME_96mDate = tME_SOMDate+relativedelta(months= -95)+relativedelta(days= -1)
    tME_120mDate = tME_SOMDate+relativedelta(months= -119)+relativedelta(days= -1)
        
    tQE_1mDate = tQE_SOMDate+relativedelta(days= -1)
    tQE_2mDate = tQE_SOMDate+relativedelta(months= -1)+relativedelta(days= -1)
    tQE_3mDate = tQE_SOMDate+relativedelta(months= -2)+relativedelta(days= -1)
    tQE_6mDate = tQE_SOMDate+relativedelta(months= -5)+relativedelta(days= -1)
    tQE_12mDate = tQE_SOMDate+relativedelta(months= -11)+relativedelta(days= -1)
    tQE_24mDate = tQE_SOMDate+relativedelta(months= -23)+relativedelta(days= -1) 
    tQE_36mDate = tQE_SOMDate+relativedelta(months= -35)+relativedelta(days= -1)
    tQE_48mDate = tQE_SOMDate+relativedelta(months= -47)+relativedelta(days= -1) 
    tQE_60mDate = tQE_SOMDate+relativedelta(months= -59)+relativedelta(days= -1) 
    tQE_72mDate = tQE_SOMDate+relativedelta(months= -71)+relativedelta(days= -1) 
    tQE_84mDate = tQE_SOMDate+relativedelta(months= -83)+relativedelta(days= -1)
    tQE_96mDate = tQE_SOMDate+relativedelta(months= -95)+relativedelta(days= -1)
    tQE_120mDate = tQE_SOMDate+relativedelta(months= -119)+relativedelta(days= -1)

    # Store Date Dataframe dates
    df_t_Dates = pd.DataFrame(columns=['Date', 'Name'])
    if t_StartDate < t_EndDate: df_t_Dates.loc[len(df_t_Dates)] = [t_EndDate,'End Date']
    if t_StartDate < tME_EndDate: df_t_Dates.loc[len(df_t_Dates)] = [tME_EndDate,'Current Month To Date']
    if t_StartDate < tQE_EndDate: df_t_Dates.loc[len(df_t_Dates)] = [tQE_EndDate,'Current Quarter To Date']
    if t_StartDate < t_CYDate: df_t_Dates.loc[len(df_t_Dates)] = [t_CYDate,'Calendar Year To Date']
    if t_StartDate < t_FYDate: df_t_Dates.loc[len(df_t_Dates)] = [t_FYDate,'Financial Year To Date']
    df_t_Dates.loc[len(df_t_Dates)] = [t_StartDate,'Since Inception']

    df_tME_Dates = pd.DataFrame(columns=['Date', 'Name'])
    if t_StartDate < tME_EndDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_EndDate,'End Date']
    if t_StartDate < tME_1mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_1mDate,'1 Month']
    if t_StartDate < tME_2mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_2mDate,'2 Months']
    if t_StartDate < tME_3mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_3mDate,'3 Months']
    if t_StartDate < tME_6mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_6mDate,'6 Months']
    if t_StartDate < tME_12mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_12mDate,'1 Year']
    if t_StartDate < tME_24mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_24mDate,'2 Years (p.a.)']
    if t_StartDate < tME_36mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_36mDate,'3 Years (p.a.)']
    if t_StartDate < tME_48mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_48mDate,'4 Years (p.a.)']
    if t_StartDate < tME_60mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_60mDate,'5 Years (p.a.)']
    if t_StartDate < tME_72mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_72mDate,'6 Years (p.a.)']
    if t_StartDate < tME_84mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_84mDate,'7 Years (p.a.)']
    if t_StartDate < tME_96mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_96mDate,'8 Years (p.a.)']
    if t_StartDate < tME_120mDate: df_tME_Dates.loc[len(df_tME_Dates)] = [tME_120mDate,'10 Years (p.a.)']
    df_tME_Dates.loc[len(df_tME_Dates)] = [t_StartDate,'Since Inception']
 
    df_tQE_Dates = pd.DataFrame(columns=['Date', 'Name'])
    if t_StartDate < tQE_EndDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_EndDate,'End Date']
    if t_StartDate < tQE_1mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_1mDate,'1 Month']
    if t_StartDate < tQE_2mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_2mDate,'2 Months']
    if t_StartDate < tQE_3mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_3mDate,'3 Months']
    if t_StartDate < tQE_6mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_6mDate,'6 Month']
    if t_StartDate < tQE_12mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_12mDate,'1 Year']
    if t_StartDate < tQE_24mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_24mDate,'2 Years (p.a.)']
    if t_StartDate < tQE_36mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_36mDate,'3 Years (p.a.)']
    if t_StartDate < tQE_48mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_48mDate,'4 Years (p.a.)']
    if t_StartDate < tQE_60mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_60mDate,'5 Years (p.a.)']
    if t_StartDate < tQE_72mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_72mDate,'6 Years (p.a.)']
    if t_StartDate < tQE_84mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_84mDate,'7 Years (p.a.)']
    if t_StartDate < tQE_96mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_96mDate,'8 Years (p.a.)']
    if t_StartDate < tQE_120mDate: df_tQE_Dates.loc[len(df_tQE_Dates)] = [tQE_120mDate,'10 Years (p.a.)']
    df_tQE_Dates.loc[len(df_tQE_Dates)] = [t_StartDate,'Since Inception']    

    return df_t_Dates, df_tME_Dates, df_tQE_Dates


# In[11]:


# MAIN LOOP Test Sandbox

testPortfolio = 'ATC70A3'
#testPortfolio = 'BON032'
t_StartDate = dt.date(2019, 6, 30)
t_EndDate = dt.date(2023, 8, 2)


# Save Attribution Grouping Names
groupName = df_BM_G1.columns[0]
groupList = df_BM_G1[df_BM_G1.columns[0]].unique()

groupListChartLabels = []
for n in range(len(groupList)):
    groupListChartLabels.append('P_'+groupName+'_'+groupList[n])


# Create Portfolio Weights

df_L1_w, df_L2_w, df_L3_w, df_Peer_w, df_Obj_w = f_portfolioWeightSets(df_validWeightFile, validSets, testPortfolio, 
                                                      df_BM_G1, t_StartDate, t_EndDate)


df_Peer_r, df_Peer_contrib = f_addPortfolioReturnGroup(df_orderedProductReturns_d, df_Peer_w, 
                                                   df_BM_G1, df_productList, 
                                                   t_StartDate, t_EndDate)

df_Obj_r, df_Obj_contrib = f_addPortfolioReturnGroup(df_orderedProductReturns_d, df_Obj_w, 
                                                   df_BM_G1, df_productList, 
                                                   t_StartDate, t_EndDate)

df_L1_r, df_L1_contrib = f_addPortfolioReturnGroup(df_orderedProductReturns_d, df_L1_w, 
                                                   df_BM_G1, df_productList, 
                                                   t_StartDate, t_EndDate)

df_L2_r, df_L2_contrib = f_addPortfolioReturnGroup(df_orderedProductReturns_d, df_L2_w, 
                                                   df_BM_G1, df_productList, 
                                                   t_StartDate, t_EndDate)

df_L3_r, df_L3_contrib = f_addPortfolioReturnGroup(df_orderedProductReturns_d, df_L3_w, 
                                                   df_BM_G1, df_productList, 
                                                   t_StartDate, t_EndDate)

df_L1_w["Peer_TOTAL"] = 0
df_L2_w["Peer_TOTAL"] = 0
df_L3_w["Peer_TOTAL"] = 0
df_L1_w["Obj_TOTAL"] = 0
df_L2_w["Obj_TOTAL"] = 0
df_L3_w["Obj_TOTAL"] = 0
df_L1_r["Peer_TOTAL"] = df_Peer_r["P_TOTAL"]
df_L2_r["Peer_TOTAL"] = df_Peer_r["P_TOTAL"]
df_L3_r["Peer_TOTAL"] = df_Peer_r["P_TOTAL"]
df_L1_r["Obj_TOTAL"] = df_Obj_r["P_TOTAL"]
df_L2_r["Obj_TOTAL"] = df_Obj_r["P_TOTAL"]
df_L3_r["Obj_TOTAL"] = df_Obj_r["P_TOTAL"]
del  df_Peer_w, df_Obj_w, df_Peer_r, df_Peer_contrib, df_Obj_r, df_Obj_contrib


# Create Portfolio Relative Level Weights
df_L2vsL1_relw = df_L2_w - df_L1_w
df_L3vsL2_relw = df_L3_w - df_L2_w

# Create Portfolio Attributions - 2 Factor Brinson Method
df_L3_2FAttrib, test1, test2 = f_2FactorBrinsonFachlerFrame(df_L3_r, df_L3_w, df_L2_r, df_L2_w, df_BM_G1, df_productList)
df_L3_1FAttrib, test1, test2 = f_2FactorBrinsonFachlerFrame(df_L3_r, df_L3_w, df_L1_r, df_L1_w, df_BM_G1, df_productList)

# Create Portfolio - Multi-period Performance Tables
t_dates, tME_dates, tQE_dates = f_SetDates(t_StartDate, t_EndDate)

# Create Limit Dataframe
df_G1_Limits = (df_rawLimitFile, df_BM_G1, testPortfolio)


# In[12]:


df_L2_w[(df_L2_w.index >= df_L2_w.index.min()) & (df_L2_w.index <= df_L2_w.index.max())]


# In[13]:


# ------------------------------------------------------------------------------
# App layout

app.layout = html.Div([
    html.H1("Atchison Test Board", style={'text-align': 'center'}),
    
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=df_L2_w.index.min(),
        end_date=df_L2_w.index.max(),
        display_format='YYYY-MM-DD',    
    ),
    
    dcc.Graph(
        id='stacked-bar-chart'
    )
])


# Define callback to update the chart based on selected date range
@app.callback(
    Output('stacked-bar-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)



def update_chart(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df_L2_w[(df_L2_w.index >= start_date) & (df_L2_w.index <= end_date)]

    y_columns = [
    'P_'+groupName+'_'+groupList[0],
    'P_'+groupName+'_'+groupList[1],
    'P_'+groupName+'_'+groupList[2],
    'P_'+groupName+'_'+groupList[3],
    'P_'+groupName+'_'+groupList[4],
    'P_'+groupName+'_'+groupList[5],
    'P_'+groupName+'_'+groupList[6]]
    
    
    fig = px.bar(
        filtered_df,
        x=filtered_df.index,
        y=y_columns, 
        title="Stacked Bar Chart",
        labels={"x": "Date", "y": "Values"},
        barmode='stack'
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)



# ------------------------------------------------------------------------------


# <br>
# 
# <h1 style="color:#93F205;"> EXAMPLE REPORT OUTPUT ---  </h1>
# 
# <br>
# <h3 style="color:#1DC8F2;"> Portfolio Asset Allocation: </h3>
# <br>
# <p style="color:#E7EAEB;">Enter some text in here..........
# </p>
# <br>

# fig_stack_L3relw = df_L3vsL2_relw.loc[t_StartDate:t_EndDate, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
#                'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]].iplot(kind='bar',barmode='relative', 
#             theme='white', title="Portfolio L3 TACTICAL Relative to Strategic Weights", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Portfolio Weight (%)")

# #printList = for n in range(len(groupList)):
# #    'P_'+groupName+'_'+groupList
# 
# fig_stack_L3w = df_L2_w.loc[t_StartDate:t_EndDate, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
#                'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]].iplot(kind='bar',barmode='stack', theme='white', title="Portfolio L3 TACTICAL Weights", 
#                                                                                                                        xaxis_title="Time (daily frequency)", yaxis_title="Portfolio Weight (%)")

# 
# <br><br>
# <h3 style="color:#1DC8F2;"> Performance: </h3>
# 
# <br>
# <p style="color:#E7EAEB;">Enter some text in here..........
# </p>
# <br>
# 
# 

# fig_bar_L3PeriodReturns_t = (f_CalcReturnTable(df_L3_r.loc[:,['P_TOTAL','BM_G1_TOTAL','Peer_TOTAL','Obj_TOTAL']], t_dates)*100).T.iplot(kind='bar', 
#         yaxis_title="Return (%, %p.a.)", title=('Total Portfolio Performance - as at Last Price '+t_dates.loc[0,'Date'].strftime("(%d %b %Y)")))

# fig_bar_L3PeriodReturns_tME = (f_CalcReturnTable(df_L3_r.loc[:,['P_TOTAL','BM_G1_TOTAL','Peer_TOTAL','Obj_TOTAL']], tME_dates)*100).T.iplot(kind='bar', yaxis_title="Return (%, %p.a.)", title=('Total Portfolio Performance - as at Last Month End '+tME_dates.loc[0,'Date'].strftime("(%d %b %Y)")))

# fig_bar_L3PeriodReturns_tQE = (f_CalcReturnTable(df_L3_r.loc[:,['P_TOTAL','BM_G1_TOTAL','Peer_TOTAL','Obj_TOTAL']], tQE_dates)*100).T.iplot(kind='bar', yaxis_title="Return (%, %p.a.)", title=('Total Portfolio Performance - as at Last Quarter End '+tQE_dates.loc[0,'Date'].strftime("(%d %b %Y)")))

# fig_line_L3Total_CumReturns = (((df_L3_r.loc[:,['P_TOTAL','BM_G1_TOTAL','Peer_TOTAL','Obj_TOTAL']] + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', title="Portfolio L3 TACTICAL Total Returns", xaxis_title="Time (daily frequency)", yaxis_title="Total Cummulative Return (%)")

# fig_line_L3G1_CumReturns = (((df_L3_r.loc[t_StartDate:t_EndDate,["P_G1_Australian Shares",
#                             "P_G1_International Shares", "P_G1_Real Assets", "P_G1_Alternatives",
#                             "P_G1_Long Duration", "P_G1_Short Duration",
#                             "P_G1_Cash"]] + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', title="Portfolio L3 TACTICAL Total Returns", xaxis_title="Time (daily frequency)", yaxis_title="Total Cummulative Return (%)")

# 
# <br>
# <h3 style="color:#1DC8F2;"> Attribution (drivers of Value Add): </h3>
# 
# 
# <br>
# <p style="color:#E7EAEB;">Enter some text in here..........
# </p>
# <br>
# 
# <p style="color:#E7EAEB;">And more in here..........
# </p>

# fig_line_L3Total_2FA = (((df_L3_1FAttrib.loc[t_StartDate:t_EndDate,['P_TOTAL_G1 -- Allocation Effect',
#                                                 'P_TOTAL_G1 -- Selection Effect']] + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', 
#                     title="Portfolio Attribution Analysis vs Reference Portfolio", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Value Add Returns (%)")
# 
# 

# (((df_L3_1FAttrib.loc[t_StartDate:t_EndDate,['G1_Australian Shares-- Allocation Effect',
#                                                 'G1_Australian Shares-- Selection Effect',
#                                                  'G1_International Shares-- Allocation Effect',
#                                                 'G1_International Shares-- Selection Effect']] + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', 
#                     title="Portfolio Attribution Analysis vs Reference Portfolio (Equity Components)", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Value Add Returns (%)")

# (((df_L3_1FAttrib.loc[t_StartDate:t_EndDate,['G1_Real Assets-- Allocation Effect',
#                                                 'G1_Real Assets-- Selection Effect',
#                                             'G1_Alternatives-- Allocation Effect',
#                                                 'G1_Alternatives-- Selection Effect']] + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', 
#                     title="L3 SAA to TAA Attribution Analysis (Alternatives)", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Value Add Returns (%)")

# (((df_L3_1FAttrib.loc[t_StartDate:t_EndDate,['G1_Long Duration-- Allocation Effect',
#                                                 'G1_Long Duration-- Selection Effect',
#                                             'G1_Short Duration-- Allocation Effect',
#                                                 'G1_Short Duration-- Selection Effect',
#                                              'G1_Cash-- Allocation Effect',
#                                             'G1_Cash-- Selection Effect']] + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', 
#                     title="L3 SAA to TAA Attribution Analysis (Defensives)", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Value Add Returns (%)")

# 
# <br>
# <h3 style="color:#1DC8F2;"> Contribution (Winners / losers): </h3>
# 
# <br>
# <p style="color:#E7EAEB;">Enter some text in here..........
# </p>
# <br>
# 
# <p style="color:#E7EAEB;">And more in here..........
# </p>

# (((df_L3_contrib.loc[t_StartDate:t_EndDate,["P_G1_Australian Shares",
#                             "P_G1_International Shares", "P_G1_Real Assets",
#                             "P_G1_Long Duration", "P_G1_Short Duration",
#                             "P_G1_Cash"]] + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', 
#                     title="Portfolio L3 TACTICAL G1 Component Weighted Contributions", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Contribution (%)")

# fig_line_L3All_CumReturns = (((df_L3_r + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', title="Cummulative Returns of All Underlying Holdings, Benchmarks and Asset Class Groupings", xaxis_title="Time (daily frequency)", 
#                                                                       yaxis_title="Constituent Product Returns (%)")

# <br><br>
# <br>
# <h3 style="color:#1DC8F2;"> APPENDIX - CHART TEST AREA.... More stuff </h3>
# <br>
# 

# #printList = for n in range(len(groupList)):
# #    'P_'+groupName+'_'+groupList
# 
# df_temp = df_L3_w.loc[t_StartDate:t_EndDate, ['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5]]].T
# df_temp['Labels'] = df_temp.index
# temp_labels = df_temp.columns
# fig_pie_L3w = df_temp.iplot(kind='pie', theme='white', title="Portfolio L3 TACTICAL Weights", 
#               labels='Labels', values=t_EndDate, hole=0.1)

# fig_bar_L3AllPeriodReturns_tME = (f_CalcReturnTable(df_L3_r.loc[:,["ANT0002AU"]], tME_dates)*100).iplot(kind='bar', title="Portfolio L3 TACTICAL Period Returns",
#                                                                                          yaxis_title="Return (%, %p.a.)")

# fig_bar_L3AllPeriodReturns_tME = (f_CalcReturnTable(df_L3_r.loc[:,:], tME_dates)*100).iplot(kind='bar', title="Portfolio L3 TACTICAL Period Returns",
#                                                                                          yaxis_title="Return (%, %p.a.)")

# fig_bar_L3AllPeriodReturns__tQE = (f_CalcReturnTable(df_L3_r.loc[:,:], tQE_dates)*100).iplot(kind='bar', title="Portfolio L3 TACTICAL Period Returns", 
#                                                                                          yaxis_title="Return (%, %p.a.)")

# #fig_bar_L3AllPeriodReturns__t = (f_CalcReturnTable(df_L3_r.loc[:,:], t_dates)*100).iplot(kind='bar', title="Portfolio L3 TACTICAL Period Returns", 
#                                                                                          yaxis_title="Return (%, %p.a.)")

# df_L3_w.loc[t_StartDate:t_EndDate, df_L3_w.columns != 'sum_TOTAL'].iplot(kind='bar',barmode='stack', 
#             theme='white', title="Portfolio L3 TACTICAL Weights", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Portfolio Weight (%)")

# (df_L3_r*100).iplot(kind='line', theme='white', 
#                     title="Product Daily Returns", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Returns (%)")

# (((df_L3_r + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', 
#                     title="L3 Tactical Product Cummulative Returns", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Returns (%)")

# (((df_L3_contrib + 1).cumprod() - 1)*100).iplot(kind='line', theme='white', 
#                     title="L3 Tactical Product Cummulative Returns", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Cummulative Returns (%)")

# df_L3vsL2_relw.loc[t_StartDate:t_EndDate,["P_G1_Australian Shares",
#                             "P_G1_International Shares", "P_G1_Real Assets",
#                             "P_G1_Long Duration", "P_G1_Short Duration",
#                             "P_G1_Cash"]].iplot(kind='bar', theme='white', 
#                     title="G1 L3 Tactical vs L2 Strategic Relative Allocation Weight", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Overweight / Underweight (%)")

# (((df_L3_r.loc[:,['P_TOTAL','Peer_TOTAL','Obj_TOTAL']] + 1).cumprod() - 1)*100).iplot(kind='line', theme='white',
#                     title="Portfolio L3 TACTICAL Total Returns", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Total Cummulative Return (%)")

# fig_stack_L3relw2 = df_L3vsL2_relw.loc[[t_StartDate,t_EndDate],['P_'+groupName+'_'+groupList[0],'P_'+groupName+'_'+groupList[1],'P_'+groupName+'_'+groupList[2],
#                'P_'+groupName+'_'+groupList[3],'P_'+groupName+'_'+groupList[4],'P_'+groupName+'_'+groupList[5],'P_'+groupName+'_'+groupList[6]]].T.iplot(kind='bar', theme='white', title="Portfolio L3 TACTICAL Relative to Strategic Weights - Difference Between 2 Dates", 
#                                                                                              xaxis_title="Time (daily frequency)", yaxis_title="Portfolio Weight (%)")

# ((df_L3_contrib.loc[t_StartDate:t_EndDate,["P_G1_Australian Shares",
#                             "P_G1_International Shares", "P_G1_Real Assets",
#                             "P_G1_Long Duration", "P_G1_Short Duration",
#                             "P_G1_Cash"]] + 1).cumprod() - 1).iplot(kind='line', theme='white', 
#                     title="Portfolio L3 TACTICAL G1 Component Contributions", 
#             xaxis_title="Time (daily frequency)", yaxis_title="Constituent Product Returns")

# In[14]:


## ADDITIONAL ITEMS TO WORK ON

# - Check BM add to 100
# - Shift allocations and make floating vs fixed option
# - fill missing returns with benchmark
# dash version

# Why Reference is 0??


# <br>
# 
# <h1 style="color:#93F205;">Appendix: Data Input Warnings / Errors</h1>
# 
# <br>
# 
# Warning Type 1 - Identify Any Missing Product Codes:

# In[15]:


err_missingProductCodes


# <br>
# 
# Warning Type 2 - Identify Any Weight Sets That Are Invalid (dont = 100%):

# In[16]:


err_invalidSet


# <br>
# 
# Warning Type 3 - Do Performance and Product Inputs Match:

# In[17]:


err_CodesMatch

