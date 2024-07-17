#!/usr/bin/env python
# coding: utf-8

# TASK-1: TITANIC SURVIVAL PREDICTION

# # IMPORTING LIBRARIES 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # #IMPORTING DATASETS

# #Collecting  the datasets

# In[2]:


import pandas as pd
df= pd.read_csv("Titanic-Dataset.csv")


# In[3]:


df


# In[4]:


#print first five rows
df.head()


# In[5]:


#print last 5 rows of datasets
df.tail()


# #DATA CLEANING

# In[6]:


df.isnull().sum()


# In[7]:


df.copy()


# #Explore the datasets

# In[8]:


#information about datasets
df.info()


# In[9]:


df.corr()


# In[10]:


#number of rows and columns in the datasets
df.shape


# In[11]:


df.replace


# In[12]:


#statistical measures about data
df.describe()


# In[13]:


df["Sex"]


# In[14]:


df['Sex'].value_counts()


# #Data visualization

# In[15]:


sns.countplot(x=df['Sex'] , hue=df['Survived'])


# In[16]:


df['Survived'].value_counts()


# In[17]:


#let's visulize the count of survivals wrt pclass
sns.countplot(x=df['Survived'],hue=df['Pclass'])



# In[18]:


#Survival rate by sex
df.groupby('Sex') [['Survived']].mean()


# In[19]:


df['Sex'] , ['Sirvived']


# In[20]:


df.isna().sum()


# In[21]:


df_final=df
df_final.head(20)


# In[22]:


import seaborn as sns
sns.countplot('Pclass' , hue='Survived' , data=df)


# In[23]:


sns.countplot('Pclass' , hue='Survived' , data=df)


# In[24]:


sns.pairplot(df, hue='Pclass')


# In[25]:


df['Sex'].value_counts()


# In[26]:


import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
plt.bar(list(df['Sex'].value_counts().keys()),list(df['Sex'].value_counts()),color='Gray')
plt.show()


# In[27]:


plt.figure(figsize=(5,8))
plt.hist(df['Age'])
plt.title('Distribution of Age')
plt.xlabel("Age")
plt.show()


# In[28]:


from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df['Sex'] = labelencoder.fit_transform(df['Sex'])

df.head()

# male-1 female-0


# #Model training

# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


# In[30]:


# Step 1: Handle missing values for 'Age', 'Embarked', and 'Fare'
# For simplicity, let's use median for 'Age' and 'Fare', and most frequent for 'Embarked'
imputer_median = SimpleImputer(strategy='median')
imputer_most_frequent = SimpleImputer(strategy='most_frequent')

df['Age'] = imputer_median.fit_transform(df[['Age']])
df['Fare'] = imputer_median.fit_transform(df[['Fare']])
df['Embarked'] = imputer_most_frequent.fit_transform(df[['Embarked']])

# Step 2: Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 3: Drop irrelevant columns (for simplicity, we use a subset of features)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Step 4: Define features (X) and target (y)
X = df.drop(columns='Survived')
y = df['Survived']


# In[31]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[33]:


model = LogisticRegression(max_iter=1000)  # Increasing max_iter to ensure convergence
model.fit(X_train, y_train)


# #Model prediction

# In[34]:


y_pred = print(model.predict(X_test))


# In[35]:


print(X_test)


# In[36]:


import warnings
warnings.filterwarnings("ignore")

def user_input():
    try:
        p_class = int(input("Enter Passenger Class (1, 2, or 3): "))
        gender = input("Enter Gender of the passenger(male or female): ").strip().lower()

        if gender == 'male':
            gender = 1
        elif gender == 'female':
            gender = 0
        else:
            raise ValueError("Invalid gender entered. Please enter 'male' or 'female'.")

        return [p_class, gender]
    
    except ValueError as e:
        print(f"Invalid input: {e}")
        return user_input()

user_input = user_input()

res = model.predict([user_input])
if res[0] == 0:
    print("So Sorry, Passenger not Survived!!")
else:
    print("Congratulations, Passenger Survived!!")


# In[ ]:




