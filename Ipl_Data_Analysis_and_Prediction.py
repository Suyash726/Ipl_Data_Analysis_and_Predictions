#!/usr/bin/env python
# coding: utf-8

# # Import the required modules#

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the csv file and getting a brief info #

# In[2]:


Ipl_data = pd.read_csv('Desktop/matches.csv')
Ipl_data.info()


# In[3]:


Ipl_data.isnull().sum()
#We can analyze the columns we need to work on


# In[4]:


Ipl_data['umpire3'].isnull().all()


# In[5]:


Ipl_data.drop('umpire3',axis=1,inplace=True)


# In[8]:


Ipl_data['winner'].fillna('Aba',inplace = True)
#Basically this imples that if a match did not have a winner then it must have been Abandoned for some reason


# In[10]:


Ipl_data['city'] = Ipl_data['city'].fillna('CapeTown')
Ipl_data.isnull().sum()


# In[11]:


#Once we are done with most of our data we need to move on the next phase i.e. Data Preparation


# #  Data Visualization and Preparation #

# In[13]:


Ipl_data.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS2','KTK','PW','RPS'],inplace=True)


Ipl_data.head(2)
#Just to get a more clear view of the data


# In[14]:


Ipl_data['toss_winner'].hist(bins = 30)


# In[17]:


Ipl_data['winner'].hist(bins = 30)


# In[18]:


temp1 = Ipl_data['toss_winner'].value_counts(sort = True)
temp2 = Ipl_data['winner'].value_counts(sort = True)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('toss_winner')
ax1.set_ylabel('Count of toss winners')
ax1.set_title("toss winners")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('winner')
ax2.set_ylabel('Count of match winners')
ax2.set_title("Winners")

#We can observe that Toss plays a really important role in winning a match. A fact that we will be analyzing in more detail


# ## Preparation ## 

# In[19]:



encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'RPS2':14},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'RPS2':14},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'RPS2':14},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'RPS2':14,'Draw':15}}

Ipl_data.replace(encode,inplace =True)

#We are giving each team a number so that it is easier for us to perform data computation


# In[34]:


#Here we are maintaining a dictionary for future reference of Teams
dicVal = encode['winner']
print(dicVal['MI']) #key value
print(list(dicVal.keys())[list(dicVal.values()).index(1)])


# In[21]:


Ipl_data = Ipl_data[['team1','team2','city','toss_decision','toss_winner','venue','winner']]
Ipl_data.head(5)
#We are representing the data in a more compact form.Taking the more important factors into consideration!


# In[24]:


df = pd.DataFrame(Ipl_data)


# In[23]:


#Data Preprocessing and Converting the Categorical data into numerical data
from sklearn.preprocessing import LabelEncoder
var = ['city','toss_decision','venue']
LE = LabelEncoder()
for i in var:
    df[i] = LE.fit_transform(df[i])
df


# In[27]:


sns.pairplot(df,hue='toss_decision')
#There are a lot of facts that can be observed in the paiplot below Like Which team chooses batting or bowling at a certain venue
#Or which decision is more likely to favor a team at certain venue


# In[47]:


mapplot = df.pivot_table(index = 'toss_decision',values='city',columns='winner')
sns.heatmap(mapplot)
#A heatmap showing how different teams performed at different cities based on toss decision


# In[37]:


sns.boxplot(x='winner',y='city',data = df)
#What we know from this graph is that there are certain cities which are more beneficial for teams . These are mostly the home venues for the Teams


# # Developing a predictive model #

# In[49]:


from sklearn.model_selection import train_test_split


# In[92]:


from sklearn import metrics
#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
      model.fit(data[predictors],data[outcome])
      predictions = model.predict(data[predictors])
      print(predictions)
      accuracy = metrics.accuracy_score(predictions,data[outcome])
      print('Accuracy : %s' % '{0:.3%}'.format(accuracy))


# In[93]:


#applying knn algorithm
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2)
classification_model(model, df,predictor_var,outcome_var)


# In[95]:


y_train=['winner']
x_train = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
model =LogisticRegression()
classification_model(model, df,x_train,y_train)


# In[109]:


#Support Vector Machine
from sklearn.svm import SVC
y_train=['winner']
x_train = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
model =SVC()
classification_model(model, df,x_train,y_train)
#Here we can see that the accuracy is close to 79% , But we can actually try using the best parameters using gridsearch


# In[108]:


from sklearn.model_selection import GridSearchCV
grid_param = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),grid_param,refit=True,verbose=3)


# In[ ]:





# In[115]:


from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='entropy')
y_train=['winner']
x_train = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,x_train,y_train)


# In[116]:


model = RandomForestClassifier(n_estimators=100)
y_train=['winner']
x_train = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,x_train,y_train)


# In[ ]:


# So we can oberve that Random Forest Classifier gives correct prediction for around 90% of the times!


# In[ ]:





# In[ ]:





# In[ ]:




