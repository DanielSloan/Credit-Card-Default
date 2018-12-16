#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('C:\\Users\\Danie\\OneDrive - University of Strathclyde\\Big Data Fundamentals\\Credit Card.csv')


#Hiding warning messages for improved presentation
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data.head()


# In[3]:


print ('Number of rows and columns ',data.shape)
data.info()


# In[4]:


#Calculating the number of defaulters
data['default'].value_counts()


# In[5]:


# Calculating the number of defaults as a percentage

#Total number of values in default variable
Total_Count = data['default'].count()

#Total number of defaults in default variable
Total_Default_Count = data[data['default']=='Yes'].default.count()

Default_Ratio = float(Total_Default_Count)/Total_Count 

print ('Percentage of defaults :',round(Default_Ratio * 100.0 , 2),'%')

# Plotting default variable
sns.countplot(x='default',data=data)


# In[6]:


# Calculating the number of students in the dataset

# Plotting student variable
sns.countplot(x='student',data=data)

#Total number of values in student variable
Total_Count = data['student'].count()

#Total number of students in the student variable
Total_Student_Count = data[data['student']=='Yes'].student.count()

Student_Ratio = float(Total_Student_Count)/ Total_Count 

print ('Percentage of students :',round(Student_Ratio * 100,2),'%')


# In[7]:


# Checking if students are less or more likely to default

Student_Data = data[data['student'] == 'Yes']
Student_Defaulters = Student_Data[Student_Data.default == 'Yes'].default.count()

print('Percentage of Students who are defaulter : ',
      round(float(Student_Defaulters)*100 /Student_Data ['student'].count(),2),'%')

Non_Student_Data = data[data['student'] == 'No']
Non_Student_Defaulters = Student_Data[Student_Data.default == 'Yes'].default.count()

print ('Percentage of Non Students who are defaulters: ', 
       round(float(Non_Student_Defaulters)*100 /Non_Student_Data['student'].count(),2), '%')


# In[8]:


title_font = {'fontname':'Arial', 'size':'18'} 
axis_font = {'fontname':'Arial', 'size':'16'} 

sns.set(style="white") #Background
sns.set(style="white", color_codes=True)

fig, ax = plt.subplots(1, 1, figsize = (12, 8)) #Setting the width and height of plot as 10 and 6 

Student_plot = plt.scatter(data.balance[data.student == 'Yes'], 
                           data.income[data.student == 'Yes'],color='b',linewidth=0.1, alpha=1)

Non_Student_plot = plt.scatter(data.balance[data.student == 'No'], 
                               data.income[data.student == 'No'],color='r',linewidth=0.1, alpha=1)


plt.legend((Student_plot, Non_Student_plot),('Students', 'Non-Students'),scatterpoints=1,loc='Upper right',ncol=1, fontsize=20,frameon=True)


plt.title("Balance and Income Distribution of Students and Non-Students",**title_font) 

plt.xlabel("Balance",**axis_font)

plt.ylabel("Income",**axis_font) 

plt.xlim(min(data.balance), max(data.balance)) #Limiting x axis to minimum and maximum value of x (i.e balance)

plt.ylim(min(data.income), max(data.income)) #Limiting y axis to minimum and maximum value of y (i.e income)


# In[9]:


title_font = {'fontname':'Arial', 'size':'18'}
axis_font = {'fontname':'Arial', 'size':'16'} 

sns.set(style="white")  # Background
sns.set(style="white", color_codes=True)

fig, ax = plt.subplots(1, 1, figsize = (12, 8)) # Setting the width and height of plot as 10 and 6 respectively

Defaulter_Plot = plt.scatter(data.balance[data.default == 'Yes'], 
                           data.income[data.default == 'Yes'],color='b',linewidth=0.1, alpha=1)

Non_Defaulter_Plot = plt.scatter(data.balance[data.default == 'No'], 
                               data.income[data.default == 'No'],color='r',linewidth=0.1, alpha=1)


plt.legend((Defaulter_Plot, Non_Defaulter_Plot),('Default', 'No Default'),scatterpoints=1,loc='Upper right',ncol=1, fontsize=20,frameon=True)

plt.title("Balance and Income Distribution of Defaulters and No Default",**title_font) 

plt.xlabel("Balance",**axis_font) 

plt.ylabel("Income",**axis_font) 

plt.xlim(min(data.balance), max(data.balance)) # Limiting x axis to minimum and maximum value 

plt.ylim(min(data.income), max(data.income)) # Limiting y axis to minimum and maximum value 


# In[10]:


data.balance.describe()


# In[11]:


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1, 1, figsize = (10, 6)) # Setting the width and height of plot as 10 and 6 
sns.boxplot(y="balance", x = 'student', data=data)


# In[12]:


fig, ax = plt.subplots(1, 1, figsize = (10, 6)) # Setting the width and height of plot as 12 and 8 

sns.boxplot(y="income", x = 'student', data=data)


# In[13]:


fig, ax = plt.subplots(1, 1, figsize = (10, 6)) # Setting the width and height of plot as 12 and 8 
sns.boxplot(y="income", x = 'default', data=data)


# In[14]:


data.groupby('default').mean()


# In[15]:


data.groupby('student').mean()


# In[16]:


# Numerical statistics for non-students
Non_Student_Stats = data[data.student=='No'].describe()
Non_Student_Stats


# In[17]:


# Numerical statistics for Default
Default_Stats = data[data.default=='Yes'].describe()
Default_Stats


# # K-Prototype Clustering

# In[18]:


from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from matplotlib import style
import csv


# In[19]:


data.head()


# In[20]:


# Importing data into an array
# Leaving out the 'default' feature before clustering - this is the dependent variable
with open ('C:\\Users\\Danie\\OneDrive - University of Strathclyde\\Big Data Fundamentals\\Credit Card.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
X= np.asarray(data)
X = X[:,1:]
data


# In[21]:


#Removing header 
X = X[1:]
X


# In[22]:


#Converting all the numerical entries into float
X[:, 1:] = X[:, 1:].astype(float)
X


# In[23]:


#Scaling Data
from sklearn import preprocessing
X[:,1] = preprocessing.scale(X[:,1])
X[:,2] = preprocessing.scale(X[:,2])
X


# In[24]:


#Running K-Prototype on dataset with different  number of clusters each time and observing the cost

Cluster_Count = []
Model_Cost = []

for k in range(2, 10):
    print ('Number of clusters : ',k)
    kproto = KPrototypes(n_clusters=k, init='Cao', verbose=2)
    clusters = kproto.fit_predict(X, categorical=[0])
    Cluster_Count.append(k)
    Model_Cost.append(kproto.cost_)
    print ('*'*100)
    


# In[25]:


x = np.array(Cluster_Count)
y= np.array(Model_Cost)
plt.plot(x, y, color='green', marker='o', linestyle='dashed',linewidth=2, markersize=10)
plt.xlabel("Number of clusters") 
plt.ylabel("Model Cost") 
plt.show()


# In[26]:


#Setting clusters to 4 as this is where the biggest decline in cost happens but we could chose any number up to 8
kproto = KPrototypes(n_clusters=4, init='Cao', verbose=2)
clusters = kproto.fit_predict(X, categorical=[0])


# In[27]:


# Print cluster centroids of the trained model
print ('Centroids are: ',kproto.cluster_centroids_)
# Print training statistics
print ('Model Cost: ',kproto.cost_)


# In[28]:


#Joining the clusters in original data
Clustered_data=pd.DataFrame({'student':X[:,0],'balance':X[:,1],'income':X[:,2],'cluster':clusters}) 
Clustered_data.head()


# In[29]:


# Dummifying student variable

Clustered_data.student[Clustered_data.student == 'Yes'] = 1
Clustered_data.student[Clustered_data.student == 'No'] = 0

Clustered_data.head()


# In[30]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig, ax = plt.subplots(1, 1, figsize = (13, 10)) # Setting the width and height of plot as 13 and 10 respectively
ax = fig.add_subplot(111, projection='3d')

x = np.array(Clustered_data['balance'])
y = np.array(Clustered_data['income'])
z = np.array(Clustered_data['student'])

x = x.astype(np.float)
y = y.astype(np.float)
z = z.astype(np.float)

ax.scatter(x,y,z, marker="s", c=clusters, s=40, cmap="viridis")

plt.show()


# # Logistic Regression 

# In[31]:


from sklearn import preprocessing
from sklearn import metrics
plt.rc("font", size=20)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
#Hiding warning messages for improved presentation
import warnings
warnings.filterwarnings('ignore')


# In[32]:


data = pd.read_csv('C:\\Users\\Danie\\OneDrive - University of Strathclyde\\Big Data Fundamentals\\Credit Card.csv')

data.head()


# In[33]:


# Dummifying default variable
data.default[data.default == 'Yes'] = 1
data.default[data.default == 'No'] = 0

# Dummifying student variable
data.student[data.student == 'Yes'] = 1
data.student[data.student == 'No'] = 0

data.head()


# In[34]:


#First logistic regression will be applied to the imbalanced dataset
#We will include income even though the exploration found it has little impact on the dependent variable default


# In[35]:


# Fitting the model using all 3 available features and predicting probabiility of default

x = data.drop('default',axis=1) # Taking out the predictors in x variable

y = data['default'] # Taking out the dependent variable in y variable

y= y.astype('int') # Need to make sure has integar format

# Splitting the data into training and test datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) # Keeping 30% of records aside as test
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

# Predicting the defaulters for test dataset and calculating the accuracy
y_pred = logreg.predict(x_test) # Contains the predictions of being defaulter or not for each record in test dataset

#logreg score compares values of dependent variable ('default') predicted by model with actual values in test dataset
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[36]:


# Using SMOTE (Synthetic Minority Oversampling Technique) to increase the number of minority classes in 
# dataset by oversampling it

from imblearn.over_sampling import SMOTE

x = data.drop('default',axis = 1)

y = data['default'] # Taking out the dependent variable in y variable
y= y.astype('int')

os = SMOTE(random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0) # Splitting the data into test and train
columns = x_train.columns

os_data_X,os_data_y=os.fit_sample(x_train, y_train) # Oversampling training dataset
os_data_X = pd.DataFrame(data=os_data_X,columns=columns ) # Converting oversampled predictors data into dataframe
os_data_y= pd.DataFrame(data=os_data_y,columns=['y']) # Converting oversampled dependent variable into dataframe

# We can check the numbers of our data
print("Length of oversampled data is ",len(os_data_X))
print("Number of non-defaulters in oversampled data ",len(os_data_y[os_data_y['y']==0]))
print("Number of defaulters",len(os_data_y[os_data_y['y']==1]))
print("Proportion of non-defaulters data in oversampled data is ",len(os_data_y[os_data_y['y']==0])*1.0/len(os_data_X))
print("Proportion of defaulters in oversampled data is ",len(os_data_y[os_data_y['y']==1])*1.0/len(os_data_X))

logreg = LogisticRegression()
logreg.fit(os_data_X,os_data_y)

from sklearn.metrics import classification_report
# Predicting the defaulters for test dataset and calculating the accuracy
y_pred = logreg.predict(x_test) # Contains the predictions of being defaulter or not for each record in test dataset on basis of 0.5


#logreg score compares values of dependent variable ('default variable') predicted by model with actual values in test dataset
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

print(classification_report(y_test, y_pred))
probs = logreg.predict_proba(x_test)[:,1]


# In[37]:


# Appending predicting values in x_test data
x_test['probs'] = pd.DataFrame(data=probs,index=x_test.index) 

x_test.head()


# In[38]:


title_font = {'fontname':'Arial', 'size':'18'}
axis_font = {'fontname':'Arial', 'size':'16'}

sns.set(style="dark")  
sns.set(style="darkgrid", color_codes=True)

fig, ax = plt.subplots(1, 1, figsize = (13, 10)) # Setting the width and height of plot as 12 and 8 

plota = plt.scatter(x_test.balance[x_test.student == 1], x_test.income[x_test.student == 1],
                    marker ='*',c=x_test[x_test.student == 1].probs, cmap='viridis',linewidth=0.1, alpha=1,s=100)

plotb = plt.scatter(x_test.balance[x_test.student == 0], x_test.income[x_test.student == 0],
                   marker ='^',c=x_test[x_test.student == 0].probs, cmap='viridis',linewidth=0.1, alpha=1,s=100)

plt.legend((plota,plotb),('Student', 'Non-Student'),scatterpoints=1,loc='Upper right',ncol=1, fontsize=20,frameon=True)


plt.title("Defaulter Probability Distribution",**title_font) 

plt.xlabel("Balance",**axis_font) 

plt.ylabel("Income",**axis_font)

plt.xlim(min(x_test.balance), max(x_test.balance)) 

plt.ylim(min(x_test.income), max(x_test.income)) 

plt.colorbar(label='Defaulter Probability')


# In[39]:


x_test.head()


# In[40]:


#Getting details for report
import sys
print (sys.version)


# In[41]:


#Getting locally imported modules from current notebook
import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

      
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name
imports = list(set(get_imports()))

requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))

