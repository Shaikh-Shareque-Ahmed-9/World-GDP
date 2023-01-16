#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from scipy import stats
import plotly.express as px
import plotly.graph_objects as go


# In[3]:


df = pd.read_csv("World GDP Dataset.csv",index_col=False)


# In[4]:


print(df.shape)
print(df.columns)
df.head()


# In[5]:


df.rename(columns = {'GDP, current prices (Billions of U.S. dollars)':'GDP'}, inplace = True)


# In[6]:


print(df.shape)
print(df.duplicated().any())
df.info()


# In[7]:


df = df.dropna()


# In[8]:


df = df.replace([0.000000], min(filter(lambda x: x > 0, df["1980"])))


# In[9]:


df = df.transpose()
df.describe()


# In[10]:


df2 = pd.DataFrame(df.values[1:], columns=df.iloc[0])
df2


# In[11]:


list(df2)


# In[12]:


easternEurope = ["Belarus","Bulgaria","Czech Republic","Hungary","Moldova","Poland","Romania","Russian Federation","Slovak Republic","Ukraine"]

sns.set(rc={'figure.figsize':(18.7,8.27)})
sns.lineplot(data = df2.loc[:,easternEurope]).set(title='Eastern Europe GDP')


# In[13]:


westernEurope = ["Germany","France","Netherlands","Belgium","Austria","Switzerland","Luxembourg"]

sns.set(rc={'figure.figsize':(18.7,8.27)})
sns.lineplot(data = df2.loc[:,westernEurope]).set(title='Western Europe GDP')


# In[14]:


northernEurope = ["United Kingdom","Sweden","Denmark","Finland","Norway","Ireland","Lithuania","Latvia","Estonia","Iceland"]

sns.set(rc={'figure.figsize':(18.7,8.27)})
sns.lineplot(data = df2.loc[:,northernEurope]).set(title='Northern Europe GDP')


# In[15]:


southernEurope = ["Italy","Spain","Greece","Portugal","Serbia","Croatia","Bosnia and Herzegovina","Albania","North Macedonia ","Slovenia","Montenegro","Malta","Andorra","San Marino"]

sns.set(rc={'figure.figsize':(18.7,8.27)})
sns.lineplot(data = df2.loc[:,southernEurope]).set(title='Southern Europe GDP')


# In[16]:


topEuropeTenGdp = ["Germany","United Kingdom","France","Italy","Russian Federation","Spain","Netherlands","Switzerland","Poland","Sweden"]

sns.set(rc={'figure.figsize':(18.7,8.27)})
sns.lineplot(data = df2.loc[:,topEuropeTenGdp]).set(title='Top GDP in Europe')


# In[17]:


topTenGdp = ["United States","China, People's Republic of","Japan","Germany","United Kingdom","India","France","Italy","Canada","Korea, Republic of"]

sns.set(rc={'figure.figsize':(18.7,8.27)})
sns.lineplot(data = df2.loc[:,topTenGdp]).set(title='Top GDP in the World')


# In[18]:


df = df.transpose()


# In[19]:


df.set_index('GDP', inplace=True, drop=True)
df


# In[20]:


df.describe()


# In[21]:


from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1]
y = df.iloc[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[22]:


X, y


# In[23]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)


# In[24]:


print(regressor.coef_)


# In[25]:


y_pred = regressor.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})


# In[26]:


print(df_preds)


# In[ ]:





# # Gross Domestic Product (GDP) 
# 
# ## Introduction
# 
# Gross Domestic Product (GDP) is a measure of the economic activity of a country. It is the total value of all goods and services produced within a country over a specific period of time, typically a year. GDP is often used as an indicator of the economic health and growth of a country.
# 
# A higher GDP generally indicates a higher level of economic activity and a stronger economy. It means that more goods and services are being produced and consumed, which can lead to more jobs and higher incomes for people living in the country. A higher GDP also typically means that the government has more resources to invest in social programs, infrastructure, and other public goods.
# 
# However, a higher GDP does not necessarily mean that the overall well-being of the population is better. GDP is only one indicator of a country's economic and social progress, and it does not take into account factors such as income inequality, environmental sustainability, or social well-being.
# 
# Additionally, a higher GDP may not be the best indicator of the well-being of the citizens, as it doesn't consider the distribution of wealth, income or access to opportunities. A country could have a high GDP but also high levels of poverty, inequality and social unrest.
# 
# For these reasons, it's important to use GDP in conjunction with other indicators, such as the Human Development Index (HDI), the Gini coefficient (next project), or the Multidimensional Poverty Index (MPI), to get a more complete picture of a country's economic and social progress.
# 

# In[27]:


df = pd.read_csv('World GDP Dataset.csv',index_col='GDP, current prices (Billions of U.S. dollars)')


# In[28]:


#Read the dataset
df2 = pd.read_csv('final.csv')
df = df.rename_axis('Country')


# In[29]:


print(df.head())
print(df2.head())


# In[30]:


# Selecting south american countries
south_america = ['Chile','Brazil','Argentina','Peru','Bolivia','Uruguay','Paraguay','Ecuador','Colombia','Venezuela']
south_america_gdp = df.loc[df.index.isin(south_america)]

south_america_gdp


# ## Which countries are above the continental average?
# 
# Now that we understand the importance of GDP as a measure of a country's economic performance, we will take a closer look at the GDP data of South American countries. We will begin by checking for any missing values in the data, as well as identifying any outliers that may skew our analysis. The following code will show the number of missing values in the data and display a box plot to visualize the distribution of GDP values for each country.
# 
# It is important to note that economic performance is not solely determined by GDP. Other factors such as population, purchasing power, and economic policies also play a significant role in a country's overall economic well-being. In order to fully understand the GDP data of these South American countries, it is crucial to consider the broader economic context of the region and the individual countries. For example, a country's GDP may be affected by events such as natural disasters, political instability, or changes in trade policies. 
# 
# By understanding the broader economic context and considering other factors, we can gain a more comprehensive understanding of a country's economic performance.

# In[31]:


# Check NA's
south_america_gdp.isna().sum()

# Check for outliers
# Create a box plot of all the columns in the DataFrame
south_america_gdp.plot(kind='box', figsize=(30,10))
plt.show()


# In[32]:


# Calculate the interquartile range (IQR)
q1 = south_america_gdp.quantile(0.25)
q3 = south_america_gdp.quantile(0.75)
iqr = q3 - q1

# Show how many times a country is classified as "outlier"
outliers = south_america_gdp.where((south_america_gdp < (q1 - 1.5 * iqr)) | (south_america_gdp > (q3 + 1.5 * iqr))).notnull()
outliers = outliers.groupby(outliers.index).sum()
outliers_total = outliers.sum(axis=1)
print(outliers_total)


# In[33]:


south_america_gdpp = df2.loc[df2.index.isin(south_america)]
#south_america_gdpp = df2.drop('region', axis=1, inplace=True)
south_america_gdpp


# # Analyzing Economic and Social Indicators in South American Countries
# 
# In this next section, we will take a closer look at the GDP of South American countries and how it compares to other economic and social indicators. First, we will start by visualizing the raw GDP data for the year 2022. This will give us a general understanding of the relative economic strength of each country. However, raw GDP data can be misleading as it does not take into account the population size of each country. Therefore, we will also adjust the GDP data to account for population size, to get a more accurate picture of the relative economic strength of each country.
# 
# In addition to GDP, we will also look at other indicators that can provide insight into the overall well-being of a country. We will look at the Health Index, which measures the overall health of a country's population, the IQ rate, which measures the average intelligence of a country's population, the GPI (Genuine Progress Indicator), which measures overall well-being and sustainable development, and finally, the IPC (Corruption Perception Index), which measures the level of perceived corruption in a country. By comparing the GDP data with these other indicators, we can get a more comprehensive understanding of the economic and social situation of each South American country.

# In[34]:



colors_dict = {'Brazil': 'yellow', 'Argentina': 'blue', 'Chile': 'red', 'Peru': 'brown','Bolivia':'green','Colombia':'orange','Ecuador':'pink','Paraguay':'gray','Uruguay':'cyan','Venezuela':'purple'}
data = []

for country in south_america_gdp.index:
    trace = go.Scatter(x=south_america_gdp.columns, y=south_america_gdp.loc[country], name=country, line=dict(color=colors_dict.get(country, 'blue'), width=2))
    data.append(trace)

layout = go.Layout(title='GDP of South American Countries over time', xaxis=dict(title='Year'), yaxis=dict(title='GDP'))

fig = go.Figure(data=data, layout=layout)
fig.update_layout(width=700, height=600)
fig.show()


# In[35]:


# Select data for the year 2022
south_america_gdp_2022 = south_america_gdp.loc[:, '2022']



# Calculate the total GDP of the continent for the year 2022
total_gdp_2022 = south_america_gdp_2022.sum()

# Divide the GDP of each country by the total GDP of the continent for the year 2022
gdp_percentage_2022 = (south_america_gdp_2022/total_gdp_2022)*100
gdp_percentage_2022 = gdp_percentage_2022.sort_values(ascending=False) 

colors = [colors_dict.get(c, 'blue') for c in gdp_percentage_2022.index]
# Create a bar chart
fig = go.Figure(data=[go.Bar(x=gdp_percentage_2022.values, y=gdp_percentage_2022.index,orientation='h', marker_color=colors)])
fig.update_layout(title_text='% of the continental GDP per country for the year 2022', xaxis_title='% of the continental GDP', yaxis_title='Country')
fig.update_layout(width=700, height=600)
fig.show()

