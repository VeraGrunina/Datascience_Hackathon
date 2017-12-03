
# coding: utf-8

# In[3]:


import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

def clear_body(text):
        text = re.sub(r"<code>(.*?(\n))+.*?</code>", "", text)
        text = re.sub(r"&(\w)*;", "", text)
        text = re.sub(r"[\d`'.,:=;!\-?\"]+", " ", text)
        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"[\s]{2,}", " ", text)
        new_text = ""
        for word in text.split():
            if word not in nltk.corpus.stopwords.words('english'):
                new_text += word + " "
        return new_text
    
def tfidf_array(corpus):
    tf = TfidfVectorizer(analyzer='word', min_df=0, stop_words='english')

    tfidf_matrix = tf.fit_transform(corpus)
    feature_names = tf.get_feature_names()

    return [tfidf_matrix.todense(), feature_names]


def get_tfidf(dense, feature_names, doc_ind, word):
    episode = dense[doc_ind].tolist()[0]
    phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    for pair in phrase_scores:
        if word == feature_names[pair[0]]:
            return pair[1]
    return 0

