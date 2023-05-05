#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
#IMPORTING THE LIBRARIES REQUIRED


# In[2]:


df = pd.read_csv("train.csv")
#IMPORTING THE CSV FILE


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


#HAVING A LIST OF NULL FEATURES

null_features = [features for features in df.columns if df[features].isnull().sum() >0]


# In[6]:


null_features


# In[7]:


#CALCULATING THE NUMBER OF NULL VALUES IN EACH FEATURES

for features in null_features:
    print(features,":",df[features].isnull().sum())


# In[8]:


#SEEING THE RELATION BETWEEN SALESPRICE I.E. DEPENDENT VARIABLE AND FEATURES WITH NULL VALUES TO CHECK IN CASE 
#THEY PLAY A MAJOR ROLE IN DETERMINING THE SALEPRICE.

for features in null_features:
    data = df.copy()
    data[features] = np.where(data[features].isnull(),1,0)
    
    data.groupby(features)['SalePrice'].median().plot.bar()
    plt.title(features)
    plt.show()


# In[9]:


#CREATING A LIST OF NUMERICAL FEATURES 
#NON-CATEGORICAL DATA DOESNT HAVE ITS DATA TYPE IDENTIFIED AS O

num_features = [features for features in df.columns if df[features].dtype != 'O']
df[num_features].head()


# In[10]:


#NUMERICAL FEATURES ALSO CONTAINS TEMPORAL VARIABLES SUCH AS DATETIME

year_features = [features for features in num_features if 'Yr'in features or 'Year'in features]
df[year_features].head()


# In[11]:


#WE GROUP THE SALEPRICE MEDIAN VALUES ACCORDING TO THE YEAR IN WHICH THE HOUSES ARE SOLD

a = df.groupby(['YearBuilt'])['SalePrice'].median()


# In[12]:


a.first


# In[13]:


#VARIATION IN THE SALEPRICE OF HOUSES YEARWISE

a.plot()


# In[14]:


#GROUPING THE SALEPRICE VALUES ACCORDING TO THE YEAR-SOLD

b = df.groupby(['YrSold'])['SalePrice'].median()
b.first


# In[15]:


b.plot()
plt.ylabel('SalePrice')
plt.title('price vs yrsold')


# We can see that the SalePrice is decreasing with the increasing years which isn't making sense. So we try to take in consideration the relationship between year-sold and year built.

# In[16]:


for feature in year_features:
    if feature != 'YrSold':
        data = df.copy()
        data[feature] = data['YrSold'] - data[feature]
        data.groupby(feature)['SalePrice'].median().plot()
        plt.show()


# Here, we can see the relation between the salesprice and the difference between yrsold and other temporal variables.
# We see that as the difference increases, theprice of the house goes down which is expected generally.

# In[17]:


for feature in year_features:
    if feature != 'YrSold':
        data = df.copy()
        data[feature] = data['YrSold'] - data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.show()


# We see the same relation with the scatter-plots also.

# In[18]:


for feature in ["YearBuilt",'YearRemodAdd','GarageYrBlt']:
    df[feature] = df['YrSold'] - df[feature]
    


# In[19]:


df.head()


# In[20]:


#CREATING THE LIST OF DISCRETE FEATURES IN THE DATAFRAME

discrete_features = [features for features in df if len(df[features].unique())<25 and features not in year_features+['Id']]
discrete_features


# In[21]:


df[discrete_features]


# In[22]:


#PLOTTING THE RELATION BETWEEN DISCRETE FEATURES AND SALEPRICE 

for feature in discrete_features:
    print(feature)
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.show()


# In[23]:


#CREATING A LIST OF CONTINUOUS FEATURES FROM THE DATAFRAME

continuous_features = [feature for feature in num_features if feature not in discrete_features and feature not in year_features +['Id']]
continuous_features


# In[24]:


#PLOTTING THE DISTRIBUTION OF CONTINUOUS FEATURES TO SEE IF ANY OF THE FEATURE HAS A SKEWED DISTRIBUTION.

#IF THERE IS A SKEWED DISTRIBUTION THEN WE CAN APPLY SCALING AND REMOVE OUTLIERS TO MAKE IT MORE LIKE A GAUSSIAN DISTRIBUTION.

for feature in continuous_features:
    data = df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.show()


# In[25]:


for feature in continuous_features:
    plt.scatter(df[feature],df['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    
    plt.show()


# In[26]:


for feature in continuous_features:
    data = df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        data[feature].hist(bins =25)
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()


# In[27]:


#DATA PLOTTING AFTER CONVERTING INTO LOG-NORMAL DISTRIBUTION


for feature in continuous_features:
    data = df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[28]:


#PLOTTING THE BOXPLOTS TO SEE THE OUTLIERS IN THE DISTRIBUTION OF CONTINUOUS FEATURES IN THE DATAFRAME

for feature in continuous_features:
    data = df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        #data['SalePrice'] = np.log(data['SalePrice'])
        data.boxplot(column = feature)
        plt.show()


# In[29]:


#CREATING THE LIST OF CATEGORICAL FEATURES

categorical_features = [feature for feature in df.columns if df[feature].dtype =='O']
categorical_features


# In[30]:


df[categorical_features].head()


# In[31]:


for feature in categorical_features:
    print(feature," --> ",df[feature].unique())
    print('\n')


# In[32]:


#PLOTTING TO SEE THE RELATION BETWEEN CATEGORICAL FEATURES AND SALEPRICE

for feature in categorical_features:
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.show()


# In[33]:


#LIST TO DEFINE CATEGORICAL FEATURES HAVING NAN VALUE

nan_categorical = [feature for feature in categorical_features if df[feature].isnull().sum()>0]
for feature in nan_categorical:
    print(feature , ' ---> ',df[feature].isnull().sum())


# In[34]:


#FUNCTION TO REPLACE NULL CATEFORICAL FEATURE VALUES TO "MISSING"

def replace_null_cat(df,nan_categorical):
    data =df.copy()
    for feature in nan_categorical:
        data[feature]= data[feature].fillna('Missing')
    return data
c = replace_null_cat(df,nan_categorical)


# In[35]:


c[nan_categorical].isnull().sum()


# In[36]:


df = c


# In[37]:


df.head()


# In[38]:


#LIST TO DEFINE NUMERICAL FEATURES WITH NAN VALUES

nan_numerical =[feature for feature in num_features if df[feature].isnull().sum()>0]
nan_numerical


# In[39]:


#FUNCTION TO DEFINE NEW COLUMNS FOR NAN VALUES AND REPLACING THE NAN VALUE WITH MEDIAN OF THE FEATURE 

def replace_null_num(df,nan_numerical):
    data = df.copy()
    for feature in nan_numerical:
        data[feature+'nan']= np.where(data[feature].isnull(),1,0)
        data[feature].fillna(data[feature].median(),inplace =True)
    return data
        


# In[40]:


d = replace_null_num(df,nan_numerical)


# In[41]:


for feature in d.columns:
    if 'nan' in feature:
        print(feature)


# In[42]:


d.shape


# In[43]:


df = d


# In[44]:


#WE ARE CONVERTING THE CONTINUOUS FEATURES INTO LOG-NORMAL DISTRIBUTION

for feature in continuous_features:
    if 0 in df[feature].unique():
        pass
    else:
        df[feature] = np.log(df[feature])
        df[feature].hist(bins =25)
        plt.xlabel(feature)
        plt.show()


# In[45]:


df.head()


# In[46]:


# CHECKING THE NO OF OCCURENCES OF EACH LABEL IN EVERY CATEGORICAL FEATURE SO THAT THE FEATURES THAT ARE VERY LESS CAN BE IGNORED
# WE USE THE GROUPBY.COUNT() FUNCTION HERE WHICH DOESN'T TAKE NaN VALUES.
for feature in categorical_features:
    a = df.groupby(feature)["SalePrice"].count()
    print(feature," --> ",a)
    print("\n")
len(df)


# In[47]:


# HERE WE USE THE GROUPBY.SIZE() FUNCTION WHICH TAKES INTO CONSIDERATION THE NaN VALUES.

for feature in categorical_features:
    a = df.groupby(feature)['SalePrice']
    print(a.size())


# In[48]:


for feature in categorical_features:
    print(df[feature].index)


# In[49]:


for feature in categorical_features:
    # FOR EVERY FEATURE, COUNT THE OCCURANCES OF FEATURE AND DIVIDE IT BY LENGTH OF COLUMN.
    a= df.groupby(feature)['SalePrice'].count() / len(df)
    # b IS A DATAFRAME WHERE THE ELEMNTS HAVE THE INDEX VALUE ONLY WHEN THE CONTRIBUTION OF FEATURE LABELS IS GREATER THAN 1%.
    b = a[a>0.01].index
    #REPLACE THE LABELS WHERE THE CONTRIBUTION IS LESS THAN 1%.
    df[feature] = np.where(df[feature].isin(b),df[feature],'Infrequent')
    


# In[50]:


df.head(50)


# Now we have to convert the categorical data into numerical encoding. Here, we will use Label Encoding.

# In[51]:


#FIRST, WE WILL CONVERT THE CATEGORICAL FEATURES INTO CATEGORY TYPE.

for feature in categorical_features:
    df[feature] = df[feature].astype('category')


# In[52]:


#SELECTING THE CATEGORY-TYPE FEATURES INTO A COLUMN SERIES

cat_features = df.select_dtypes(['category']).columns


# In[53]:


#APPLYING THE X.CAT.CODES FUNCTION TO PERFORM THE LABEL ENCODING TO EVERY FEATURE COLUMN.

df[cat_features] = df[cat_features].apply(lambda x: x.cat.codes)


# In[54]:


df.head()


# In[55]:


#FEATURE SCALING
scaling_feature = [feature for feature in df.columns if feature not in ['Id','SalePrice'] ] 
scaler = MinMaxScaler()
scaler.fit(df[scaling_feature])


# In[56]:


df.head()


# In[57]:


#CONCATINATING THE TWO COLUMNS--> 1) DATAFRAME CONTAINING ID AND SALEPRICE.....2) DATAFRAME CONTAINING TRANSFORMED VALUES OF SCALING FEATURE ......
#WITH AXIS =1 MEANS THEY SHOULD BE JOINED IN COLUMN FORMAT

df_train = pd.concat([df[['Id','SalePrice']].reset_index(drop=True),pd.DataFrame(scaler.transform(df[scaling_feature]),columns=scaling_feature)],axis =1)


# In[58]:


df_train.head()


# In[59]:


#EXPORTING THE DATA FILE TO CSV FORMAT.

df_train.to_csv('house_prediction_train_data.csv',index = False)


# In[ ]:




