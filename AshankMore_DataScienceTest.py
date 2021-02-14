#!/usr/bin/env python
# coding: utf-8

# # Data Science Position Research Test
#   
#   ###### Submitted By: Ashank More

# ### Data File: The Crash Reporting - Incidents Data from Montgomery County, MD 
# Source: https://catalog.data.gov/dataset/crash-reporting-incidents-data 
# Description: This dataset provides general information about each collision and details of all traffic 
# collisions occurring on county and local roadways within Montgomery County, as collected via the Automated 
# Crash Reporting System (ACRS) of the MD State Police, and reported by the Montgomery County Police, 
# Gaithersburg Police, Rockville Police, or the Maryland-National Capital Park Police.
# 

# ### Importing all the important libraries. 
# <br> Pandas: For data manipulation
# <br> Numpy: For Mathematical Computation
# <br> Seaborn, Matplotlib: For data visualization
# <br> Sklearn: For Machine Learning

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
import datetime as dt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the data
# Reading the file using Pandas. Storing the data in a dataframe 'df'.

# In[3]:


df = pd.read_csv("./Desktop/Crash_Reporting_-_Incidents_Data.csv")


# # Data Exploration
# 
# ### Knowing more about the dataset.
# 
# <br>These commands are used to take a look at specific sections of pandas DataFrame.
# <br>df.head(n): First n rows of the DataFrame
# <br>df.tail(n): Last n rows of the DataFrame
# <br>df.isnull(): Return a boolean same-sized object indicating if the values are NA.
# <br>df.shape: Number of rows and columns
# <br>df.info(): Index, Datatype and Memory information
# <br>df.value_counts(dropna=False): View unique values and counts
# <br>df.duplicated(): Returns true if there are duplicates and false otherwise

# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


duplicateRowsDF = df[df.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF)


# In[8]:


df.isnull().sum(axis = 0)


# In[11]:


df1 =df
df2 =df
df3 =df


# ### Findings: 
# <br>1) The data has too many missing values.
# <br>2) Manjority of the data is categorical.
# <br>3) There are no duplicates presesnt in the data.
# <br>4) The data comprises of 66494 rows and 45 columns

# <br>

# # Data Cleaning

# ### Cleaning I
# <br>Handling Null values: 
# <br>Using the below command we remove the data which has more than 40% Null values.
# <br>This reduces our number of columns to 36.

# In[12]:


cols = df.columns[df.isnull().mean()>0.4]
df=df.drop(cols, axis=1)


# In[17]:


df.shape


# In[18]:


df.isnull().sum(axis = 0)


# ### Cleaning II
# <br> Replacing Column Names: 
# <br> There are a lot of spaces in column names. This sometimes creates a problem. 
# <br> Using the below command we replace the space with a blank.
# <br> Column value is year, so we'll rename it.

# In[19]:


df.columns = df.columns.str.replace(' ', '') 


# In[20]:


df.rename(columns={'#VALUE!':'Year'}, inplace=True)


# # Data Wrangling

# ### Creating New Columns
# <br>In order to test the hypothesis we need to create new column using the exisiting columns.
# <br>To create a new column using conditions, we first create functions. 
# <br>Then a new column is created and values are filled using the functions. 
# <br>
# <br>This is done by using .apply method of python.
# <br>df[col_name].apply(func_name,axis=1): Applies the function func_name across each row of df[col_name]
# <br>
# <br>To verify our results we will random samples of the subset of the dataset. 
# <br>df.sample(n): displays n random rows from the dataframe df.
# <br>
# <br>To check the value count of our new column we will usedf.value_counts()

# #### New Column I : Visibility 
# <br>This will help us understand the visibility of the driver based on light conditions and weather.
# <br>The code will return High if there is ample light and weather is clear, Medium if its dusk or down and weather is unclear,
# <br>low for all other cases when weather is unclear. 
# <br>Note: Unknown conditions are considered as Medium.

# In[22]:


def Visibility(df):
    if (df['Light']==('DARK LIGHTS ON') or df['Light']==('DAYLIGHT')) and df['Weather'] == 'CLEAR':
        return 'High'
    elif df['Light'] == (('DARK NO LIGHTS') or df['Light']==('DARK -- UNKNOWN LIGHTING')) and df['Weather'] != 'CLEAR':
        return 'Low'
    else:
        return 'Medium'
    


# In[23]:


df['Visibility']=df.apply(Visibility, axis=1)


# In[25]:


df[['Visibility', 'Light', 'Weather']].sample(5)


# In[26]:


df['Visibility'].value_counts()


# #### New Column II : DUI
# <br>This will help us understand weather the person was under the influence or not.
# <br>The code will return DUI if there is any substance detected else it will return Not DUI
# <br>Note: Unknown is classified as Not DUI.

# In[27]:


def DUI(df):
    if (df['DriverSubstanceAbuse']==('NONE DETECTED') or df['DriverSubstanceAbuse']== ('NONE DETECTED, UNKNOWN') or 
    df['DriverSubstanceAbuse']== 'UNKNOWN' or  df['DriverSubstanceAbuse']=='N/A, NONE DETECTED' or 
    df['DriverSubstanceAbuse']=='N/A, UNKNOWN'  or df['DriverSubstanceAbuse']== 'NONE DETECTED, OTHER'  or                           
    df['DriverSubstanceAbuse']=='N/A, NONE DETECTED, UNKNOWN'):
        return 'Not DUI'
    else:
        return 'DUI'


# In[28]:


df['DUI']=df.apply(DUI, axis=1)


# In[32]:


df[['DUI', 'DriverSubstanceAbuse']].sample(5)


# In[33]:


df['DUI'].value_counts()


# In[34]:


df3=df
df4=df


# #### New Column III : Crashtype
# 
# <br>This will be dichotomous column. If the crash in injusry or fatal crash, then the code will return 'Severe' 
# <br>else it will retuen 'Property damage'.
# <br>Note: In this case we will changing the exisiting column (ACRS Report Type) rather than creating a new one.

# In[38]:


def Crashtype(df):
    if df['ACRSReportType']==('Property Damage Crash'):
        return 'Property Damage Crash'
    else:
        return 'Severe' 


# In[39]:


df['ACRSReportType']=df.apply(Crashtype, axis=1)


# In[40]:


df['ACRSReportType'].value_counts()


# #### New Column IV : Condition
# <br>This will help us get the main parameters for our hypothesis test. If the Visibility is low or Medium and the person is under
# <br>influence then the code will return Unfavorable. Else it will return Favorable.

# In[64]:


def CNDTN(df):
    if (df['Visibility']=='Low' or df['Visibility']=='Medium') and df['DUI'] == 'DUI':
        return 'Unfavorable'
    else:
        return 'Favorable'


# In[65]:


df['Condition'] = df.apply(CNDTN,axis=1)


# In[66]:


df[['Condition', 'ACRSReportType']].sample(5)


# In[67]:


df['Condition'].value_counts()


# ### Cleaning III
# <br> Removing Column Names: 
# <br> We will drop all the columns that are not neccessary for our analysis.

# In[45]:


df.drop(columns=['LocalCaseNumber','AgencyName','RouteType','MilePoint','MilePointDirection', 
                 'LaneNumber','Direction', 'Distance','DistanceUnit',
                 'Cross-StreetName','RoadName','Cross-StreetType'
                 ], inplace=True)
df.info()


# <br>

# # Hypothesis 1
# 
# <br>Question: Does Unfavorable Conditions (Low Visibility and DUI) affect the type of crash (Severe or Property damage) ?
# 
# <br><b>H0</b>: Unfavorable Condition are not associated with the type of crash
# <br><b>Ha</b>: Unfavorable Condition are associated with the type of crash
# 
# <br>We will be using Chi-Square Test:
# <br>The test is applied when you have two categorical variables from a single population. 
# <br>It is used to determine whether there is a significant association between the two variables.
# 

# Step 1:
# <br>We first create a Contingency table.
# <br>A Contingency table (also called crosstab) is used in statistics to summarise the relationship between several categorical <br>variables. Here, we take a table that shows the number of favorable and Unfavorable conditions leading to different types of Crashes.
# <br>Using the Contingency table we get Observed value table

# In[68]:


dataset_table=pd.crosstab(df['Condition'],df['ACRSReportType'])
print(dataset_table)


# In[69]:


#Observed Values
Observed_Values = dataset_table.values 
print("Observed Values :-\n",Observed_Values)


# Step2: Calculating the Expected value table
# <br> SciPy: used in mathematics, engineering, scientific and technical computing. 

# In[49]:


import scipy.stats as stats


# In[70]:


val=stats.chi2_contingency(dataset_table)
val


# In[71]:


Expected_Values=val[3]


# In[72]:


no_of_rows=len(dataset_table.iloc[0:2,0])
no_of_columns=len(dataset_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05


# Step 3: Performing the test using Python (scipy.stats)
# 
# <br>We define a significance factor to determine whether the relation between the variables is of considerable significance. <br>Generally a significance factor or alpha value of 0.05 is chosen. This alpha value denotes the probability of erroneously <br>rejecting H0 when it is true. A lower alpha value is chosen in cases where we expect more precision. If the p-value for the <br>test comes out to be strictly greater than the alpha value, then H0 holds true.
# 
# <br>Using chi-square value:
# <br>If our calculated value of chi-square is less or equal to the tabular(also called critical) value of chi-square, then H0 holds true.

# In[83]:


from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]


# In[84]:


print("chi-square statistic:-",chi_square_statistic)


# In[85]:


critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)


# Using chi-square statistic and degree of freedom we find p value using the below formula

# In[81]:


p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('p-value:',p_value)


# In[82]:


if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# Since, p-value > alpha. Therefore, we Reject H0, that is, the variables do have a significant relation.

# # Hypothesis 2
# 
# <br>Question: Does Injury crash lead to more hit and run cases?
# 
# <br><b>H0</b>: Injury crash has no relation to hit and run cases.
# <br><b>Ha</b>: Injury crash leads to more hit and cases.
# 
# <br>We will be using Fisher’s exact test 
# <br>Fisher's exact test is an alternative statistical significance test to chi square test used in the analysis of 2 x 2 contingency tables. 
# <br>It is one of a class of exact tests, so called because the significance of the deviation from a null hypothesis ( P-value) can be calculated exactly, rather than relying on an approximation that becomes exact as the sample size grows to infinity, as seen with chi-square test. 
# <br>It is used to examine the significance of the association between the two kinds of classification. 
# 

# Step 1: Create a table with frequencies of crash type and hit/ run

# In[59]:


ar=np.array([[176.0, 9.0],[22051.0, 1516.0]])    
nf=pd.DataFrame(ar, columns=["H/R No", "H/R Yes"])
nf.index=["Fatal Crash", "Injury crash"] 
nf


# Step 2: Perform Fisher’s Exact Test.
# 
# We perform Fisher’s Exact Test using the fisher_exact function from the SciPy library.

# In[60]:


oddsratio, pvalue = stats.fisher_exact([[176.0, 9.0],[22051.0, 1516.0]])  
pvalue


# In[61]:


fisher_pvalue = pvalue
alpha = 0.05


# In[62]:


if fisher_pvalue<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# Since this p-value is not less than 0.05, we do not reject the null hypothesis. Thus, we don’t have sufficient evidence to say that there is a significant association between Hit and run and type of crash.
