# -*- coding: utf-8 -*-

import pandas as pd

#######################################################################################

from pickle import load
df = load(open("./data/DATASET_ALL.pkl", 'rb+'))
df = load(open("./data/DATASET_PP_Part_D_1201-0111.pkl", 'rb+'))

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

template_header = ['what do i include with my appeal request', 'notice of denial of medicare',
                   'request of redetermination of medicare']
template_header = list(map(filter_stopword, template_header))

# Using TemplateClassfier - Template
from StandardForm.template import TemplateClassifier
tc  = TemplateClassifier(min_tokens_per_template=5)

tc.fit_for_template(dataset['doc_content'], template_header, tokens_p_template=25,
                    template_sep='page break ml processing',
                   max_df=0.95, min_df=100, ngram_range=(4, 4))

# Exploring template extracted
print(len(tc.templates_))
tc.templates_.keys()

# Get form_template label
tc.get_form_labels()
tc.get_form_labels(tc.templates_[119])

# Predict new Tempalte
tc.predict_template(dataset['doc_content_fil'][7864].split('page break ml processing'))  # 7864

# Save Model as pickle
tc.save_model('./pickle/TokenVectorizer_PD_t25.pkl')  # Full

# Load Model picle
tc.load_model('TokenVectorizer_dataset_all_t20.pkl', 1)

tc1  = TemplateClassifier(min_tokens_per_template=10)
tc1.load_model('tc.pkl', 1)


#######################################################################################

