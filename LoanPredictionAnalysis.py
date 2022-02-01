#!/usr/bin/env python
# coding: utf-8

# # Data Set Information

# Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.
# 
# This is a standard supervised classification task.A classification problem where we have to predict whether a loan would be approved or not. Below is the dataset attributes with description.

# 
# 
# 

# # Import Modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # Loading the dataset

# In[2]:


df = pd.read_csv("train_data.csv")
df.head()


# In[3]:


df.describe()


# In[4]:


df.shape


# In[5]:


df.info()


# # Preprocessing the dataset

# In[6]:


# find the null values
df.isnull().sum()


# In[7]:


# fill the missing values for numerical terms - mean
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


# In[8]:


# fill the missing values for categorical terms - mode
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])


# In[9]:


df.isnull().sum()


# # Exploratory Data Analysis

# In[10]:


# categorical attributes visualization
sns.countplot(df['Gender'])


# In[11]:


sns.countplot(df['Married'])


# In[12]:


sns.countplot(df['Dependents'])


# In[13]:


sns.countplot(df['Education'])


# In[14]:


sns.countplot(df['Self_Employed'])


# In[15]:


sns.countplot(df['Property_Area'])


# In[16]:


sns.countplot(df['Loan_Status'])


# In[ ]:





# In[17]:


# numerical attributes visualization
sns.distplot(df["ApplicantIncome"])


# In[18]:


sns.distplot(df["CoapplicantIncome"])


# In[19]:


sns.distplot(df["LoanAmount"])


# In[20]:


sns.distplot(df['Loan_Amount_Term'])


# In[21]:


sns.distplot(df['Credit_History'])


# In[ ]:





# # Creation of new attributes

# In[22]:


# total income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# # Log Transformation

# In[23]:


# apply log transformation to the attribute
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome']+1)
sns.distplot(df["ApplicantIncomeLog"])


# In[24]:


df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome']+1)
sns.distplot(df["CoapplicantIncomeLog"])


# In[25]:


df['LoanAmountLog'] = np.log(df['LoanAmount']+1)
sns.distplot(df["LoanAmountLog"])


# In[26]:


df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term']+1)
sns.distplot(df["Loan_Amount_Term_Log"])


# In[27]:


df['Total_Income_Log'] = np.log(df['Total_Income']+1)
sns.distplot(df["Total_Income_Log"])


# # Coorelation Matrix

# In[28]:


corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap="BuPu")


# In[29]:


df.head()


# In[30]:


# drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)
df.head()


# # Label Encoding

# In[31]:


from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# In[32]:


df.head()


# # Train-Test Split

# In[33]:


# specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# # Model Training

# In[35]:


# classify function
from sklearn.model_selection import cross_val_score
def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test)*100)
    # cross validation - it is used for better validation of model
    # eg: cv-5, train-4, test-1
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is",np.mean(score)*100)


# In[36]:



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# In[37]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)


# In[38]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
model = RandomForestClassifier()
classify(model, X, y)


# In[39]:


model = ExtraTreesClassifier()
classify(model, X, y)


# # Hyperparameter tuning

# In[40]:


model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=1)
classify(model, X, y)


# # Confusion Matrix

# A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.

# In[41]:


model = RandomForestClassifier()
model.fit(x_train, y_train)


# In[42]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[43]:


sns.heatmap(cm, annot=True)


# In[44]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[45]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[46]:


sns.heatmap(cm, annot=True)


# In[ ]:




