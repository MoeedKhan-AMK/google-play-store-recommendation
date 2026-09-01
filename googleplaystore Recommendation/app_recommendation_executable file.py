#!/usr/bin/env python
# coding: utf-8

# ## GOOGLE PLAY STORE APPS RECOMMENDATION SYSTEM
# 
# - We will use following columns ['App,
# Category,
# Type,
# Content Rating,
# Genres'] for our recommendation system.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


google_play_store = pd.read_csv('./googleplaystore.csv')
google_play_store.head(5)


# In[3]:


google_play_store.shape


# In[4]:


google_play_store.columns


# In[5]:


play_store_df = google_play_store[["App", "Category", "Type", "Content Rating", "Genres"]]
play_store_df


# In[6]:


play_store_df["Category"].value_counts()


# In[7]:


play_store_df[play_store_df['Category'] == '1.9']


# In[8]:


# DROPPED UNNECESSARY COLUMN
play_store_df.drop(play_store_df[play_store_df['Category'] == '1.9'].index, inplace=True)


# In[9]:


play_store_df.dropna(inplace=True)
play_store_df.shape


# In[10]:


# So, in our 'Genres'column, we need to do some cleaning:
# - We removed all the punctuations (&,;...)
play_store_df["Genres"]


# In[11]:


def collapse(text):
    text = text.replace("&"," ")
    text = text.replace(";"," ")
    text = text.replace("_"," ")
    text = text.replace("  "," ")
    return text.strip()


# In[12]:


play_store_df["Genres"] = play_store_df["Genres"].apply(collapse).str.lower()
play_store_df


# In[13]:


play_store_df["Category"] = play_store_df["Category"].apply(collapse).str.lower()
play_store_df["App"] = play_store_df["App"].str.lower()
play_store_df["Type"] = play_store_df["Type"].str.lower()
play_store_df["Content Rating"] = play_store_df["Content Rating"].str.lower()

play_store_df


# In[14]:


play_store_df


# In[15]:


play_store_df["tags"] = play_store_df["Category"] + " " + play_store_df["Type"] + " " + play_store_df["Content Rating"] + " "+ play_store_df["Genres"]
play_store_df


# In[16]:


final_df = play_store_df[["App","tags"]]
final_df


# In[17]:


final_df["tags"]


# #### stemming
# - NLP helps in machine understanding human language

# In[18]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[19]:


def stem(text):
    lst = []

    for i in text.split():
        lst.append(ps.stem(i))

    return " ".join(lst)


# In[20]:


final_df["tags"] = final_df["tags"].apply(stem)
final_df


# In[21]:


# Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[22]:


cv = CountVectorizer(max_features=5000)


# In[23]:


vector = cv.fit_transform(final_df["tags"])


# In[24]:


vector = vector.toarray()
vector


# In[25]:


list(cv.get_feature_names_out())


# In[26]:


final_df.shape


# In[27]:


from sklearn.metrics.pairwise import cosine_similarity


# In[73]:


similarity = cosine_similarity(vector)
similarity.shape


# In[74]:


final_df.reset_index(drop=True, inplace=True)
final_df


# In[30]:


# AS OF NOW, WE HAVE COMPLETED SOME CLEANING, STEMMING AND VECTORIZATION

# NEXT STEP IS TO REMOVE DUPLICATED VALUES

#start working on dealing with duplicate values and later on move with the plan


# In[105]:


## Recommendation step

final_df[final_df["App"] == "facebook"]


# In[32]:


final_df.duplicated().sum()


# In[33]:


final_df = final_df.drop_duplicates()


# In[34]:


final_df.duplicated().sum()


# In[ ]:





# In[35]:


final_df[final_df["App"] == "roblox"].iloc[0]


# In[36]:


final_df = final_df.drop(final_df[(final_df["App"] == "roblox") & (final_df["tags"].str.contains("famili"))].index)


# In[37]:


final_df[final_df["App"] == "netflix"]


# In[38]:


# Dropping duplicates values based on App column only
final_df = final_df.drop_duplicates(subset=["App"], keep="first")
final_df


# In[100]:


final_df[final_df["App"] == "facebook"]


# In[103]:


vector[2002]


# In[106]:


app_index = final_df[final_df["App"] == "netflix"].index
app_index


# In[54]:


app_index = final_df[final_df["App"] == "netflix"].index[0]
app_index


# In[109]:


# we will save similarity of 'netflix' movie in distances. Folling is the similarity score of 'netflix' with other vectors.

distances = similarity[app_index]
distances


# In[110]:


# By applying enumarte function, it assigned each value index number

index_list = list(enumerate(distances))
index_list


# - With this, we have synced index number in following variables 'distances, final_df, vector'

# In[111]:


# So now we wanted top 10 similar movies
# we will sort, then get 10 similar movies
# Sort function with 'reverse=True' will sort values in descending order on basis of 'index_list'. we use'key= lambda x:x[1]' to achieve this

similar_app = sorted(index_list, reverse=True,key=lambda x: x[1])
similar_app


# In[71]:


final_df.loc[691]


# In[67]:


for i in similar_app:
    print(i)


# In[129]:


for index, similarity_score in similar_app[1:11]:
    print(final_df.iloc[index, 0])


# In[127]:


def recommend(app_name):
        app_index = final_df[final_df["App"] == app_name].index[0]
        distances = similarity[app_index]

        similar_app = sorted(list(enumerate(distances)), reverse=True,key=lambda x: x[1])


        for index, similarity_score in similar_app[1:11]:
            print(final_df.iloc[index,0])



# In[128]:


recommend("facebook")


# In[97]:


## AS OF NOW, THE CODE IS WORKING BUT THERE IS A PROBLEM WITH SIMILARITY, WHICH IS WHY IT IS NOT RECOMMENDING CORRECTLY
## DEBUG THE CODE, CHECK FROM WHERE THE PROBLEM IS OCCURING


# In[ ]:




