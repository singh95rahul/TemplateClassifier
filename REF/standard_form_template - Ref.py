# -*- coding: utf-8 -*-

import pandas as pd

#######################################################################################

from pickle import load
df = load(open("./data/DATASET_ALL.pkl", 'rb+'))
df = load(open("./data/DATASET_PARTD.pkl", 'rb+'))

# Prepare Dataset
dataset = df.copy()
dataset.drop(['urgency_o', 'urgency_n', 'ignore_page', 'load_date', 'UniqueImageID', 'confidence', 'IntakeChannelQueueLkup'],
             axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['UrgencyLkup'] = le.fit_transform(dataset['UrgencyLkup'])

# Cleaning Data
from nltk import corpus
STOPWORDS = list(set(corpus.stopwords.words('english')))

def filter_stopword(sentence):
    return ' '.join([word for word in sentence.split() if word not in STOPWORDS])

dataset['doc_content_fil'] = dataset['doc_content'].apply(filter_stopword)


#######################################################################################


# Part D
template_header = ['claim reconsideration request', 'health insurance claim']

# Using TemplateClassfier - Template
from StandardForm.template import TemplateClassifier
tc = TemplateClassifier(min_tokens_per_template=5)

tc.fit_for_template(dataset['doc_content'], template_header, tokens_p_template=15,
                    template_sep='page break for ml processing',
                   max_df=0.90, min_df=0.01, ngram_range=(4, 4))

# Exploring template extracted
print(len(tc.templates_))
tc.templates_.keys()

# Predict new Tempalte
tc.predict_template(dataset['doc_content_fil'][7864].split('page break ml processing'))  # 7864

# Get form_template label
tc.get_form_labels()
tc.get_form_labels(tc.templates_[119])

# Save Model as pickle
tc.save_model('tc_0.pkl', 0)  # Partial
tc.save_model('tc_1.pkl', 1)  # Full

# Load Model picle
tc.load_model('TokenVectorizer_dataset_all_t20.pkl', 1)

tc1  = TemplateClassifier(min_tokens_per_template=10)
tc1.load_model('tc.pkl', 1)


#######################################################################################

# Using TemplateClassfier - Cluster
from StandardForm.template import TemplateClassifier
tc = TemplateClassifier(min_tokens_per_template=9)

# Finding Best Clusters
score, dist = [], []
for cls in range(2, 50):
    ret = tc.fit_for_cluster(dataset['doc_content'][:], template_sep='page break for ml processing',
                             n_clusters=cls,
                             max_df=0.90, min_df=0.01, ngram_range=(4, 4))
    score.append(ret[0])
    dist.append(ret[1])

# Visualizing cluster scores
from matplotlib import pyplot as plt

plt.plot(range(2, len(dist) + 2), dist)
plt.show();
plt.plot(range(2, len(dist) + 2), score)
plt.show();

# Fitting Best cluster based on Elbow Method
ret = tc.fit_for_cluster(dataset['doc_content'][:], template_sep='page break for ml processing',
                         n_clusters=42,
                         max_df=0.90, min_df=0.01, ngram_range=(4, 4))
print(ret)

# Get Cluster centroids
print(tc.km.cluster_centers_)

# Predict Cluster
# 0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 30, 32, 33, 34, 35, 37, 38, 39, 40 
tc.predict_cluster(dataset['doc_content_fil'][100].split('page break ml processing'))


tc.save_model('TokenVectorizer_dataset_partd_42cls.pkl', 1)  # Full

