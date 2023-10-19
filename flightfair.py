#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[2]:


train_data = pd.read_excel(r"Data_Train.xlsx")


# In[3]:


pd.set_option('display.max_columns',None)


# In[4]:


train_data.head()


# In[5]:


train_data.info()


# In[6]:


train_data["Duration"].value_counts()


# In[7]:


train_data.shape 

train_data.dropna(inplace = True)


# In[8]:


train_data.isnull().sum()


# In[9]:


train_data["Journey_day"]= pd.to_datetime(train_data.Date_of_Journey, format ="%d/%m/%Y").dt.day


# In[10]:


train_data["Journey_month"]= pd.to_datetime(train_data.Date_of_Journey, format = "%d/%m/%Y").dt.month


# In[11]:


train_data.head()


# In[12]:


train_data.head()

train_data["Dep_hour"]= pd.to_datetime(train_data.Dep_Time).dt.hour

# In[13]:


train_data["Dep_min"]= pd.to_datetime(train_data.Dep_Time).dt.minute


# In[14]:


train_data.head()


# In[15]:


train_data.drop(["Dep_Time"], axis = 1, inplace = True)


# In[16]:


train_data.head()


# In[17]:


train_data["Arrival_hour"]= pd.to_datetime(train_data.Arrival_Time).dt.hour


# In[18]:


train_data["Arrival_min"]= pd.to_datetime(train_data.Arrival_Time).dt.minute


# In[19]:


train_data.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[20]:


train_data.head()


# In[21]:


duration = list(train_data["Duration"])

duration_hours = []
duration_mins = []

for d in duration:
    parts = d.split()
    hours = 0
    mins = 0
    for part in parts:
        if 'h' in part:
            hours = int(part.replace('h', ''))
        elif 'm' in part:
            mins = int(part.replace('m', ''))
    duration_hours.append(hours)
    duration_mins.append(mins)


# In[22]:


train_data["Duration_hours"] = duration_hours


# In[23]:


train_data["Duration_mins"] = duration_mins


# In[24]:


train_data.head()


# In[25]:


train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[26]:


train_data.head()


# In[27]:


train_data.drop(["Duration"], axis =1 , inplace = True)


# In[28]:


train_data.head()


# In[29]:


train_data["Airline"].value_counts()


# In[82]:


# Sort the data
sorted_train_data = train_data.sort_values("Price", ascending=False)

# Create the plot
sns.catplot(y="Price", x="Airline", data=sorted_train_data, kind="boxen", height=6, aspect=3)

# Display the plot
plt.show()


# In[31]:


Airline = train_data[["Airline"]]

Airline = pd.get_dummies(Airline , drop_first= True)

Airline.head()


# In[32]:


train_data["Source"].value_counts()


# In[33]:


sns.catplot(y = "Price", x = "Source", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)
plt.show()


# In[34]:


Source = train_data[["Source"]]

Source = pd.get_dummies( Source, drop_first = True)

Source.head()


# In[35]:


train_data["Destination"].value_counts()


# In[36]:


Destination = train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[37]:


train_data["Route"]


# In[38]:


train_data.drop(["Route", "Additional_Info" ], axis= 1 , inplace = True)


# In[39]:


train_data["Total_Stops"].value_counts()


# In[40]:


train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[41]:


train_data.head()


# In[42]:


data_train = pd.concat([train_data, Airline,Source,Destination], axis = 1)


# In[43]:


data_train.head()


# In[44]:


data_train.drop(["Airline","Source","Destination"], axis = 1 , inplace = True)


# In[45]:


data_train.head()


# In[46]:


data_train.shape


# In[47]:


test_data = pd.read_excel(r"Test_set.xlsx")


# In[48]:


test_data.head()


# In[49]:


# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[50]:


data_test.head()


# In[51]:


data_train.shape


# In[52]:


data_train.columns


# In[53]:


X = pd.DataFrame(data_train, columns=['Total_Stops', 'Price', 'Journey_day', 'Journey_month', 'Dep_min',
                                      'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins',
                                      'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
                                      'Airline_Jet Airways', 'Airline_Jet Airways Business',
                                      'Airline_Multiple carriers',
                                      'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
                                      'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
                                      'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
                                      'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
                                      'Destination_Kolkata', 'Destination_New Delhi'])

X.head()


# In[54]:


y = data_train.iloc[:,1]
y.head()


# In[55]:


numeric_columns = train_data.select_dtypes(include=['number'])

# Create a correlation matrix for the numeric columns
correlation_matrix = numeric_columns.corr()

# Set the figure size
plt.figure(figsize=(18, 18))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn")

# Show the plot
plt.show()


# In[56]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming X and y are your feature matrix and target variable
# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in X_train and X_test
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy if needed
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create and fit your ExtraTreesRegressor
selection = ExtraTreesRegressor()  # You can also specify hyperparameters if needed
selection.fit(X_train_imputed, y_train)

# Now, you can use the trained model for prediction
y_pred = selection.predict(X_test_imputed)

# Print the predicted values
print("Predicted Values (y_pred):")
print(y_pred)

# You can also print evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R^2 Score:", r2)
print("Mean Squared Error:", mse)


# In[57]:


print(selection.feature_importances_)


# In[58]:


plt.figure(figsize=(12,8))
feat_importance = pd.Series(selection.feature_importances_, index = X.columns)
feat_importance.nlargest(20).plot(kind="barh")
plt.show()


# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[60]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Create an imputer with strategy 'mean' to fill missing values
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on your training data
X_train_imputed = imputer.fit_transform(X_train)

# Create a RandomForestRegressor model
reg_rf = RandomForestRegressor()

# Fit the model to the imputed training data
reg_rf.fit(X_train_imputed, y_train)


# In[68]:


pip install --upgrade scikit-learn


# In[69]:


pip install --upgrade pip


# In[74]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Assuming X and y_train are your dataframes or arrays with inconsistent sample sizes

# Align the dataframes to have consistent samples
X, y_train = X[:8546], y_train[:8546]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Create a RandomForestRegressor model
reg_rf = RandomForestRegressor()

# Fit the model to the imputed training data
reg_rf.fit(X_imputed, y_train)

# Assuming X_test is your test data
# Handle missing values in the test data
X_test_imputed = imputer.transform(X_test)

# Make predictions on the imputed test data
y_pred = reg_rf.predict(X_test_imputed)


# In[83]:


from sklearn.impute import SimpleImputer

# Check for NaN values in your dataset
print("NaN values in X before imputation:", np.isnan(X).sum())

# Create an imputer with strategy 'mean' to fill missing values
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on your dataset
X_imputed = imputer.fit_transform(X)

# Check for NaN values after imputation
print("NaN values in X after imputation:", np.isnan(X_imputed).sum())

# Ensure no NaN values are present
if np.isnan(X_imputed).sum() == 0:
    reg_rf.fit(X_imputed, y_train)
    print("Model successfully fitted!")

    # Evaluate model performance on the training data
    score = reg_rf.score(X_imputed, y_train)
    print(f"Model score on training data: {score * 100:.2f}%")
else:
    print("Some NaN values still exist after imputation. Please check your data.")


# In[ ]:





# In[ ]:





# In[ ]:




