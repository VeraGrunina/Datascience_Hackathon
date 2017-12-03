
# coding: utf-8

# In[ ]:


import numpy
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

def reshape_tags(dataframe, num_clusters):

    def clear_tags(series):
        return series.apply(lambda row: str(row).replace("><"," ").replace("<","").replace(">",""))

    def get_tags_corpus(series):
        return series.apply(lambda row: row.split(' ')).to_dense().tolist()

    def get_new_tags(row,tagToNew):
        list_new_tags = list(filter(lambda x: x != None, [tagToNew.get(i) for i in row.split(' ')]))
        max_tag = max(set(list_new_tags), key = list_new_tags.count,default = None)
        return max_tag     
    
    def change_tags(dataframe, tagToNew):
        dataframe['Tags'] = clear_tags(dataframe['Tags']).apply(lambda tags: get_new_tags(tags,tagToNew))
        return dataframe
        
    corpus = get_tags_corpus(clear_tags(dataframe['Tags']))

    model = Word2Vec(corpus)

    word_vectors = model.wv.syn0
    kmeans = KMeans(n_clusters=num_clusters)
    idx = kmeans.fit(word_vectors)
    
    tagToNew = {}
    for i in range(len(idx.labels_.tolist())):
        tagToNew[str(model.wv.index2word[i])]=idx.labels_[i]
        
    return change_tags(dataframe, tagToNew)

