#!/usr/bin/env python
# coding: utf-8

# # Analyze the sales of the company in order to target new areas of growth

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
import statistics


# In[2]:


transactions = pd.read_csv('transactions.csv')
print(transactions)


# In[3]:


## check to clean for bad data
transactions.describe()


# In[4]:


## Replace bad data (date starting with test)
mask = transactions['date'].str.startswith('test_')
transactions.loc[mask, 'date'] = transactions.loc[mask, 'date'].str.replace('test_', '')
print(mask)


# In[5]:


## check to clean for bad data
transactions.describe()


# In[11]:


## check to clean for bad data
mask.describe()


# In[12]:


# How many missing values in data
transactions.isnull().sum().sum()


# In[13]:


#Check missing by variable
transactions.isnull().sum()


# In[14]:


products = pd.read_csv('products.csv')
print(products)


# In[15]:


## check to clean for bad data
products.describe()


# In[16]:


## Remove negative 'price' values
cleaned_products = products[products['price'] >= 0]
print(cleaned_products)


# In[17]:


## check to clean for bad data
cleaned_products.describe()


# In[18]:


products.isnull().sum()


# In[19]:


## left join 'product' table to 'transaction' table
d = pd.merge(transactions, cleaned_products, how='left', on=['id_prod'])
print(d)


# In[20]:


d.isnull().sum()
## production data has extra id_prod values that do not exist in transaction data, therefore there are 303 null columns when
## these datasets are joined together


# In[21]:


#perform regular/inner join instead to only bring data that exists in both tables
d1 = transactions.merge(cleaned_products, left_on = 'id_prod', right_on ='id_prod')
print(d1)


# In[22]:


## check to clean for bad data
d1.describe()


# In[23]:


d1.isnull().sum()


# In[24]:


customers = pd.read_csv('customers.csv')
print(customers)


# In[25]:


## check to clean for bad data
customers.describe()


# In[26]:


customers.isnull().sum()


# In[27]:


df = d1.merge(customers, left_on = 'client_id', right_on ='client_id')
print(df)


# In[28]:


## check to clean for bad data
df.describe()


# In[29]:


df.isnull().sum()


# ##### Central tendency (Mean, Median, Mode) and Dispersion Measures (Variance and Standard Deviation)

# In[30]:


# mean
round(statistics.mean(df['price']),2)


# In[31]:


# median
statistics.median(df['price'])


# In[32]:


# mode
statistics.mode(df['price'])


# In[33]:


# Variance
round(statistics.variance(df['price']),2)


# In[29]:


# Standart Deviation
round(statistics.stdev(df['price']),2)


# ##### Lorenz curve and a Gini coefficient
# ###### are used to measure the variability of the distribution of income and wealth. Hence, Lorenz Curve is the measure of the deviation of the actual distribution of a statistical series from the line of equal distribution. The extent of this deviation is known as Lorenz Coefficient.

# In[34]:


# Sort values by 'price'
df = df.sort_values(by=['price'])
print(df)


# 

# In[35]:


# Calculate the cumulative sum of the sorted data and divide it by the total sum of the data 
df['cumulative_perc'] = df['price'].cumsum() / df['price'].sum() 
print(df)


# In[36]:


# Calculate the Gini coefficient 
area_under_curve = df['cumulative_perc'].sum() / len(df) 
area_between_curve_and_diagonal = 0.5 - area_under_curve 
gini_coefficient = area_between_curve_and_diagonal / 0.5
print(f'The Gini coefficient is {gini_coefficient:.2f}')


# In[37]:


# Generate x values
x = np.linspace(0, 1, 10)

# Define the line equation with the desired gini_coefficient
y = gini_coefficient * x

# Plot the line
##plt.plot(x, y, label='Line with Coefficient 0.39', color='blue')

plt.plot(x, y, label='gini_coefficient', color='blue')

# Add labels and legend
plt.title('gini_coefficient')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()

##Calculate the Lorenz curve 
plt.plot(np.linspace(0, 1, len(df)), df['cumulative_perc'].values)


# In[38]:


plt.plot(np.linspace(0, 1, len(df)), df['cumulative_perc'].values)
plt.plot(x, y, label='gini_coefficient', color='blue')


# #### Graphic representations, including at least one histogram, a representation with boxplots, and a time series graph (i.e a graph in which the abscissa axis represents dates).

# In[39]:


# creating a Histogram by Price
plt.hist(df['price']) 
plt.show()


# In[40]:


# creating a Histogram by sex
plt.hist(df['sex']) 
plt.show()


# In[41]:


# Boxplot sex vs price
df.boxplot(by='sex',column='price',grid= False)


# In[42]:


# Boxplot categ vs price
df.boxplot(by='categ',column='price',grid= False)


# ### Time series graph

# In[43]:


df = d1.merge(customers, left_on = 'client_id', right_on ='client_id')
print(df)


# In[44]:


df.dtypes


# In[45]:


# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])


# In[46]:


# Cluster dates into months
df['month'] = df['date'].dt.to_period('M')


# In[47]:


# Group by month and aggregate values by price
monthly_data = df.groupby('month').agg({'price': 'sum'}).reset_index()
print(monthly_data)


# In[48]:


monthly_data.dtypes


# In[49]:


monthly_data['month'] = monthly_data['month'].astype(str)  # Convert the month to string for plotting
plt.plot(monthly_data['month'], monthly_data['price'])  # Plot the 'price' column
plt.title('Monthly Price Sum')  # Set the title of the plot
plt.xlabel('Month')  # Set the label for the x-axis
plt.ylabel('Price')  # Set the label for the y-axis
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.show()  # Display the plot


# ### Bivariate analyses: Scatter plot, Correlation analysis, Regression

# In[83]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# In[85]:


# Pairplot using seaborn
sns.pairplot(df)
plt.show()


# #### Scatter plot:

# In[58]:


# Scatter plot Price vs birth year(X = Independent variable, Y = Dependent Variable)
plt.figure(figsize=(8, 6))
plt.scatter(df['birth'], df['price'])
plt.title('Scatter Plot of X vs Y')
plt.xlabel('Birth Year')
plt.ylabel('Price')
##plt.grid(True)
plt.show()


# #### Simple Linear Regression

# In[67]:


df = d1.merge(customers, left_on = 'client_id', right_on ='client_id')
print(df)


# In[68]:


# Split the dataframe into features (X) and target variable (y)
X = df[['birth']]
y = df['price']


# In[69]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[70]:


# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[71]:


# Make predictions
y_pred = model.predict(X_test)


# In[73]:


# Plot the data and the model's predictions
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Birth Year')
plt.ylabel('Price')
plt.title('Linear Regression') 


# ## Task 3

# ### Is there a correlation between gender and categories of products purchased?

# In[91]:


df = d1.merge(customers, left_on = 'client_id', right_on ='client_id')
print(df)


# In[95]:


# Compute the correlation between X and Y
correlation = df['categ'].corr(df['sex'])

print("Correlation between categ and sex:", correlation)


# In[ ]:




