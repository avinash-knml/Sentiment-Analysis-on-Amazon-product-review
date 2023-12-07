#!/usr/bin/env python
# coding: utf-8

import streamlit as st
st.header('NLP Sentiment Analysis')

st.subheader('Business Objective:')

st.write(' - Extracting sentiment from customer reviews on a product')

st.subheader('Data Collection')

st.write('* Product : Apple 2020 MacBook Air Laptop M1 chip, 13.3-inch')
st.write('* E-commerse website : www.amazon.in')

st.write(' MacBook Air was the first product in Mac series launched with the M1 chip produced by Apple itself, which replaced the Intel Prossesors in Apple products. ')
st.write(' Apple integrated the CPU, GPU, Neural Engine, I/O and so much more onto a single tiny chip. M1 delivers exceptional performance, custom technologies and unbelievable power efficiency. Introduction of M1 chip was considered as a major breakthrough for Mac.')
 
st.write(' This gives a relevence in analyzing the customer reviews about the new product from a e-commerse website.')
 
st.write(' Link : https://www.amazon.in/Apple-MacBook-Chip-13-inch-256GB/product-reviews/B08N5W4NNB/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews')

st.write("* Data is collected from an online web-srapper app")

# In[1]:

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Loading Dataset
st.subheader('Dataset')
scrap=pd.read_excel("C:\\Users\\avina\\Documents\\Data Science\\Projects\\P-212-NLP Sentiment Analysis\\macbook m1.xlsx")
scrap


# In[3]:


# Creating a df containing only review content
review_0=pd.DataFrame(scrap[('Content')])


# In[4]:


# Finding number of duplicated reviews
review_0.duplicated().sum()


# In[5]:


# Deleting the dupllicated reviews
review_1=review_0.drop_duplicates()
#review_1[review_1.duplicated()]


# In[6]:


# Finding the null rows
review_1.isnull().sum()


# In[7]:


# Deleting the null rows
review_2=review_1.dropna(axis=0)
review_2.isnull().sum()


# In[8]:


review_2.info()


# In[9]:


review_2.describe()


# In[10]:


# setting the dataframe as 'final'
review=review_2


# In[11]:


# Removing emojis from reviews
import re

# Define the emoji pattern
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F9D0-\U0001F9DF"
                           "]+", flags=re.UNICODE)


# In[12]:


# Remove emojis from the 'Content' column
review['Content'] = review['Content'].apply(lambda x: emoji_pattern.sub(r'', x))



# In[13]:


# Converting to lower case
review['Content'] = review['Content'].str.lower()
st.subheader('Final Data')
review


# In[14]:


#get_ipython().system('pip install textblob')
#get_ipython().system('pip install wordcloud')


# In[15]:


#NLTK libraries

import nltk
import re
import string
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

#Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


# Creating sentiment column

def get_sentiment_label(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'


# In[17]:


review['Sentiment'] = review['Content'].apply(get_sentiment_label)
#review


# In[18]:


counts=review['Sentiment'].value_counts()
#counts


# ### Creating word cloud for all reviews

# In[19]:


# Joining the list into one string/text
review_j=' '.join(review['Content'])
#review_j


# In[20]:


review_nop=review_j.translate(str.maketrans('','', string.punctuation))
#review_nop


# In[21]:


nltk.download('punkt')
from nltk.tokenize import word_tokenize
review_token=word_tokenize(review_nop)
#review_token


# In[22]:


#get_ipython().system('pip install stopwords')


# In[23]:


# Removing stopwords
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
stop=stopwords.words('english')
review_nosw=[word for word in review_token if not word in stop]
#review_nosw


# In[24]:


df_rev=pd.DataFrame({'rev':review_nosw})
#df_rev


# In[25]:


# concatenate the preprocessed text into a single string
review_w=' '.join(df_rev['rev'])


# In[26]:


# create the word cloud
st.subheader('Word Cloud')
wordcloud = WordCloud(width=800, height=800, background_color='black', min_font_size=10).generate(review_w)

# plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# ### Creating word cloud for positive reviews

# In[27]:

st.subheader('Word Cloud (Positive Reviews)')
positive=review[review['Sentiment']=='Positive']
#positive


# In[28]:


# Joining the list into one string/text
positive_j=''.join(positive['Content'])


# In[29]:


# Removing Punctuation
positive_nop=positive_j.translate(str.maketrans('','',string.punctuation))


# In[30]:


# Tokanization
positive_token=word_tokenize(positive_nop)


# In[31]:


# Removing stopwords
positive_nosw=[word for word in positive_token if not word in stop]


# In[32]:


df_pos=pd.DataFrame({'rev':positive_nosw})
#df_pos


# In[33]:


# concatenate the preprocessed text into a single string
positive_w=' '.join(df_pos['rev'])


# In[34]:


# create the word cloud
wordcloud_pos = WordCloud(width=800, height=800, background_color='black', min_font_size=10).generate(positive_w)

# plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_pos)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
st.pyplot()


# ### Creating word cloud for negative reviews

# In[35]:

st.subheader('Word Cloud (Negative Reviews)')
negative=review[review['Sentiment']=='Negative']
#negative


# In[36]:


# Joining the list into one string/text
negative_j=''.join(negative['Content'])


# In[37]:


# Removing Punctuation
negative_nop=negative_j.translate(str.maketrans('','',string.punctuation))


# In[38]:


# Tokanization
negative_token=word_tokenize(negative_nop)


# In[39]:


# Removing stopwords
negative_nosw=[word for word in negative_token if not word in stop]


# In[40]:


df_neg=pd.DataFrame({'rev':negative_nosw})
#df_neg


# In[41]:


# concatenate the preprocessed text into a single string
negative_w=' '.join(df_neg['rev'])

# create the word cloud
wordcloud_neg = WordCloud(width=800, height=800, background_color='black', min_font_size=10).generate(negative_w)

# plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_neg)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
st.pyplot()


# ### Creating word cloud for positive reviews

# In[42]:

st.subheader('Word Cloud (Neutral Reviews)')
neutral=review[review['Sentiment']=='Neutral']
#neutral

# Joining the list into one string/text
neutral_j=''.join(neutral['Content'])

# Removing Punctuation
neutral_nop=neutral_j.translate(str.maketrans('','',string.punctuation))

# Tokanization
neutral_token=word_tokenize(neutral_nop)

# Removing stopwords
neutral_nosw=[word for word in neutral_token if not word in stop]

df_neu=pd.DataFrame({'rev':neutral_nosw})
#df_neu

# concatenate the preprocessed text into a single string
neutral_w=' '.join(df_neu['rev'])

# create the word cloud
wordcloud_neu = WordCloud(width=800, height=800, background_color='black', min_font_size=10).generate(neutral_w)

# plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_neu)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
st.pyplot()

# ## Balancing the data set

# In[43]:


#review


# In[44]:


counts=review['Sentiment'].value_counts()
#counts


# Here we have, 
# Positive reviews = 577, 
# Negative reviews = 70, 
# Neutral reviews = 66 in numbers. 
# 
# For a better result for sentiment analysis we need to balance the data.

# In[45]:


#get_ipython().system('pip install imblearn')


# In[46]:


from imblearn.over_sampling import SMOTE
from collections import Counter


# In[47]:


for feature in review.columns: 
    if review['Content'].dtype == 'object': 
        review['Content'] = pd.Categorical(review['Content']).codes


# In[48]:


# Separate the features and labels
x = review['Content']
y = review['Sentiment']


# In[49]:


#x


# In[50]:


# Print the class distribution before applying SMOTE
#print('Class distribution before SMOTE:', Counter(y))


# In[51]:


# Apply SMOTE to balance the dataset
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(x.values.reshape(-1,1), y)


# In[52]:


#X_res


# In[53]:


#print("After balancing: ", Counter(y_res))


# In[54]:


# Convert the balanced data to a pandas DataFrame
df_balanced = pd.DataFrame({"review_text": X_res.reshape(-1), "sentiment": y_res})


# In[55]:


#df_balanced.head()


# In[56]:


df_balanced.sentiment[df_balanced.sentiment =='Neutral'] =0
df_balanced.sentiment[df_balanced.sentiment =='Positive'] =1
df_balanced.sentiment[df_balanced.sentiment =='Negative'] =2


# In[57]:


#df_balanced


# In[58]:


X = df_balanced[['review_text']]
Y = df_balanced['sentiment'].astype('int')


# ## Model Building

# **We divided data into train and test set**

# In[59]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.30,random_state=0)


# ### Random Forest Classifier Model

# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


model1 = RandomForestClassifier(n_estimators=200, random_state=0)


# In[62]:


model1.fit(xtrain,  ytrain)


# In[63]:


predictions1 = model1.predict(xtest)


# In[64]:


#predictions1


# In[65]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(ytest,predictions1))
#print(classification_report(ytest,predictions1))
#print(accuracy_score(ytest, predictions1))


# ### KNeighbours Classifier

# In[66]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors =6)#no of neighbors is hpyer parameter
model2.fit(xtrain, ytrain)


# In[67]:


predictions2 = model2.predict(xtest)


# In[68]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(ytest,predictions2))
#print(classification_report(ytest,predictions2))
#print(accuracy_score(ytest, predictions2))


# ### Logistic Regression

# In[69]:


from sklearn.linear_model import LogisticRegression
model3 =LogisticRegression()


# In[70]:


model3.fit(xtrain, ytrain)


# In[71]:


predictions3 = model3.predict(xtest)


# In[72]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(ytest,predictions3))
#print(classification_report(ytest,predictions3))
#print(accuracy_score(ytest, predictions3))


# ### DecisionTreeClassifier

# In[73]:


from sklearn.tree import DecisionTreeClassifier
model4= DecisionTreeClassifier(criterion="gini")
model4.fit(xtrain, ytrain)


# In[74]:


predictions4 = model4.predict(xtest)


# In[75]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(ytest,predictions4))
#print(classification_report(ytest,predictions4))
#print(accuracy_score(ytest, predictions4))


# In[76]:

st.subheader('Models and accuracy')
accuracy=pd.DataFrame()


# In[77]:


accuracy['Model']=('Random Forest','KNeighbour','Logistic Regreassion','Decision Tree')
accuracy['Accuracy_Score']=(accuracy_score(ytest, predictions1),accuracy_score(ytest, predictions2),
                           accuracy_score(ytest, predictions3),accuracy_score(ytest, predictions4))
accuracy


# We have KNeighbour Model with graeter accuracy than Random Forest, Logistic Regression, Decision Tree models.
# 
# So, we are fixing the KNeighbour model as our final model.

# In[169]:

st.subheader('Sentiment Prediction')
new_reviews=pd.DataFrame()
review_input = st.text_input("Type review:")
new_reviews['review'] = (review_input,)
new_reviews['sentiment']= new_reviews['review'].apply(get_sentiment_label)
new_reviews['sentiment']


# In[170]:


for feature in review.columns: 
    if new_reviews['review'].dtype == 'object': 
        new_reviews['review'] = pd.Categorical(new_reviews['review']).codes


# In[171]:


l=new_reviews['review']
#l


# In[172]:


new_reviews.sentiment[new_reviews.sentiment =='Neutral'] =0
new_reviews.sentiment[new_reviews.sentiment =='Positive'] =1
new_reviews.sentiment[new_reviews.sentiment =='Negative'] =2


# In[173]:


#new_reviews


# In[174]:


n=np.array(l)
nnn=n.reshape(-1,1)


# In[175]:

st.subheader('Model Prediction')
finalprediction = model2.predict(nnn)
if finalprediction == 0:
    st.write("Review Sentiment is: ",'Neutral 😐')
elif finalprediction == 1:
    st.write("Review Sentiment is: ",'Positive 😃')
elif finalprediction == 2:
    st.write("Review Sentiment is: ",'Negative 😕')






