
# coding: utf-8

# In[3]:


import pandas as pd
import numpy
from tags import reshape_tags
import tf_idf as ti

posts_df = pd.read_csv("Posts.csv",  encoding = "ISO-8859-1")

reshape_tags(posts_df,10)

clear_posts = posts_df['Body'].apply(lambda x: ti.clear_body(x))

tfidf = ti.tfidf_array(clear_posts)

model = Word2Vec(clear_posts.apply(lambda row: row.split()),size=200)

def get_w2v(string, keyedvectors):
    try:
        return keyedvectors[string]
    except:
        return 0

vectors = list(map(lambda i: sum([get_tfidf(tfidf[0],tfidf[1],i ,word) * get_w2v(word ,model.wv) for word in clear_posts[i].split()]),range(len(clear_posts.tolist()))))

def distance(x, y):
    return numpy.dot(x, y) / (numpy.linalg.norm(x) * numpy.linalg.norm(y))


