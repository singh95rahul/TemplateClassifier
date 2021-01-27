# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:47:06 2020

@author: rsing177
"""
from itertools import chain
from pickle import load, dump

import pandas as pd
from numpy import NaN, min, array
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score


class TemplateClassifier:
    """
    TemplateClassifier is a generic unsupervised classifier, where if finds different type of common forms among the
    data set and also classifies the templates in real time once trained.
    """
    def __init__(self, min_tokens_per_template):
        """
        Constructor for TemplateClassifier
        :param min_tokens_per_template: int; Minimum number of tokens to be extracted for each template
        """
        self.cv = None
        self.km = None
        self.MIN_TOKENS = min_tokens_per_template
        self.templates_ = None
        self.__form_template_data = None

    def __fit_cv(self, data, **kwargs):
        """
        Encapsulated method for fitting data to CountVentorizer, with kwargs.
        kwargs - support all the kwargs of sklearn.feature_extraction.text.CountVectorizer
        :param data: Pandas Series; Dataset than should be fitted to the CV
        :param save_cv: Bool: Default - False; Save the fitted CV to the template cv
        :param kwargs: Parameters of sklearn.feature_extraction.text.CountVectorizer
        :return: Transformed data
        """
        print(" |+ Fitting Vectorizer +|")
        self.cv = CountVectorizer(**kwargs)
        data = self.cv.fit_transform(data)
        print(" |+ Vectorizer fit completed. ==> ", self.cv)

        print(" |+ Extracting core Vectorizer.. ", end='')
        self.cv = CountVectorizer(**kwargs, **{'vocabulary': self.cv.vocabulary_})
        print(" Done +|")

        return data

    # Train TemplateClassifier Model for template types - Using Clusters
    def fit_for_cluster(self, data, template_sep=None, n_clusters=2, refit=False, **kwargs):
        """
        Fit the TemplateClassifier with data to extract the common templates' tokens. This method will fit the TC
        and generate token that can be used to identify different common templates among the data.
        :param data: Pd.Series or List; Dataset than should be fitted to the CV
        :param template_sep: Str; Separator to be used for splitting the data into different/smaller templates
        :param n_clusters: int; Default = 2; Number of KMeans clusters to be formed
        :param refit: Refit the Model. Ignore pre-trained cache
        :param kwargs: Parameters of sklearn.feature_extraction.text.CountVectorizer
        :return: None
        """

        # Checking params
        if not (isinstance(data, pd.Series) or isinstance(data, list)):
            raise TypeError(f"Expected type pd.Series or list, received - {type(data)}")

        if isinstance(data, list):
            data = pd.Series(data)

        if self.__form_template_data is None or refit:
            print(" |+ Analyzing Contents +|")
            if template_sep is not None:
                print(f" |+ Splitting Templates form content, using separator - '{template_sep}'.. ", end='')
                data = data.str.split(template_sep)
                data = data.apply(pd.Series).stack().reset_index()[0].replace('', NaN).dropna().reset_index(drop=True)
                print(" Done +|")

            data_trans_df = self.__fit_cv(data.copy(), **kwargs)

            # Filtering Templates that had some token
            print(" |+ Generating Standard templates.. ", end='')
            self.__form_template_data = data_trans_df[
                (data_trans_df.sum(axis=1) > int(self.MIN_TOKENS)).reshape(-1).nonzero()[-1]].toarray()
            self.__form_template_data = pd.DataFrame(self.__form_template_data)
            print(" Done +|")
        else:
            print(" |+ Using Pre-fitted cache +|")

        print(f" |+ Clustering templates for - {n_clusters} clusters +|")
        if n_clusters is not None and isinstance(n_clusters, int) and n_clusters > 1:
            self.km = KMeans(n_clusters=n_clusters)
            self.km.fit(self.__form_template_data)
            s_score = silhouette_score(self.__form_template_data, self.km.labels_)
            distance = sum(min(cdist(self.__form_template_data, self.km.cluster_centers_, 'euclidean'),
                               axis=1)) / self.__form_template_data.shape[0]
            return s_score, distance
        else:
            raise ValueError("Invalid value for n_cluster")

    # ...
    def predict_cluster(self, x):
        """
        Classify 'X' to the pre-trained clusters, based on token.
        :param x: Pd.Series or List;
        :return: Classified cluster
        """
        # Checking params
        if not hasattr(self, 'km') or self.km is None:
            raise AttributeError("TemplateClassifier object not fitted, first fit the object to use predict_cluster")
        if not hasattr(self, 'cv') or self.cv is None:
            raise AttributeError("TemplateClassifier object not fitted, first fit the object to use predict_cluster")
        if not (isinstance(x, pd.Series) or isinstance(x, list)):
            raise TypeError(f"Expected type pd.Series or list, received - {type(x)}")

        if isinstance(x, list):
            x = pd.Series(x)

        # Predict Cluster
        x_test = self.cv.transform(x)
        y_test = self.km.predict(x_test)
        pred = list(zip(x_test, y_test))

        # Get Euclidean distance from the cluster centroid
        euclidean_dist = array(list((chain(*[cdist(_x.todense(), self.km.cluster_centers_[_l:_l + 1], 'euclidean') for _x, _l in pred]))))
        return y_test, euclidean_dist.reshape(-1,)

    # Train TemplateClassifier Model for template types - Using Tokens
    def fit_for_template(self, data, template_headers, tokens_p_template, template_sep=None, **kwargs):
        """
        Fit the TemplateClassifier with data to extract the common templates' tokens. This method will fit the TC
        and generate token that can be used to identify different common templates among the data.
        :param data: Pd.Series or List; Dataset than should be fitted to the CV
        :param template_headers: List; Sample templates token. Like Form Name etc.
        :param tokens_p_template: tuple(int, int); (Minimum number of tokens to be extracted from each template,
                                                    Maximum number of tokens to be extracted from each template)
        :param template_sep: Optional - Str; Separator to be used for splitting the data into different/smaller
            templates, if data is not at template level
        :param kwargs: Parameters of sklearn.feature_extraction.text.CountVectorizer
        :return: None
        """

        # Checking params
        if not (isinstance(data, pd.Series) or isinstance(data, list)):
            raise TypeError(f"Expected type pd.Series or list, received - {type(data)}")
        if not isinstance(template_headers, list) or len(template_headers) == 0:
            raise ValueError("template_header should be of type - list, and cannot be empty")
        if not hasattr(tokens_p_template, 'index') or len(template_headers) != 2:
            raise ValueError("tokens_p_template should be of type - list/tuple of length 2")

        tokens_p_template = tuple(map(int, tokens_p_template))
        if not tokens_p_template[0] < tokens_p_template[1]:
            raise ValueError("tokens_p_template - Value at index 0 should be less than value at index 1")

        if isinstance(data, list):
            data = pd.Series(data)

        print(" |+ Analyzing Contents +|")
        data_transformed = self.__fit_cv(data.copy(), **kwargs)
        data_trans_df = pd.DataFrame(data_transformed.toarray(), columns=self.cv.get_feature_names())

        print(" |+ Extracting templates to fit..", end='')
        # Extracting similar template headers
        template_headers = list(chain(
            *[[feature for feature in self.cv.get_feature_names() if header in feature] for header in
              template_headers]))

        if not len(template_headers):
            print(f"\n !! Empty templates header - Try less specific names in template_headers !!")
            return

        # Extracting additional tokens which are part of, templates that has above headers
        # Filtering contents that has template headers
        data_trans_df = data_trans_df.loc[data_trans_df[template_headers].apply(lambda x: any(x > 0),
                                                                                axis=1)].replace(0, NaN)
        data = data.iloc[data_trans_df.dropna(thresh=self.MIN_TOKENS, axis=1).index]
        print(" Done +|")

        if not len(data):
            print(f"\n !! Empty dataset after filter - Try reducing MIN_TOKEN and refit !!")
            return

        # Separating content into individual templates to similar templates
        if template_sep is not None:
            print(f" |+ Splitting Templates form content, using separator - '{template_sep}'.. ", end='')
            data = data.str.split(template_sep.lower())
            data = data.apply(pd.Series).stack().reset_index()[0].replace('', NaN).dropna().reset_index(drop=True)
            print(" Done +|")

        data_trans_df = self.__fit_cv(data.copy(), **kwargs)

        # Filtering Templates that had some token
        print(" |+ Generating Standard templates.. ", end='')
        self.__form_template_data = data_trans_df[
            (data_trans_df.sum(axis=1) > int(self.MIN_TOKENS)).reshape(-1).nonzero()[-1]].toarray()
        self.__form_template_data = pd.DataFrame(self.__form_template_data)

        self.templates_ = dict()
        filtered_templates = []

        for form in [self.cv.vocabulary.get(e, None) for e in template_headers]:
            if form is not None:
                temp_df = self.__form_template_data.loc[self.__form_template_data[form] > 0].replace(0, NaN)
                temp_df = temp_df.dropna(thresh=self.MIN_TOKENS, axis=1)
                form_tokens = set(
                    temp_df.count(axis=0).sort_values(ascending=False)[:tokens_p_template[1]].index).union([form])
                # Filtering templates that does not meet minimum number of token criteria
                if len(form_tokens) > tokens_p_template[0]:
                    self.templates_[form] = form_tokens
                else:
                    filtered_templates.append(form)

        print(" Done +|")
        print(f"\nNote - Filtered {len(filtered_templates)} templates due to Minimum number of token condition")
        return

    def __classify_token_to_form(self, x):
        for (label, _template) in self.templates_.items():
            if len(set(x).intersection(_template)) >= self.MIN_TOKENS:
                return label
        return -1

    # Predict template type
    def predict_template(self, x):
        """
        Classify 'X' to the best template match, based on token.
        :param x: Pd.Series or List;
        :return: Classified template
        """

        # Checking params
        if not hasattr(self, 'templates_') or self.templates_ is None:
            raise AttributeError("Vocabulary not fitted or provided")
        if not hasattr(self, 'cv') or self.cv is None:
            raise AttributeError("Vocabulary not fitted or provided")
        if not (isinstance(x, pd.Series) or isinstance(x, list)):
            raise TypeError(f"Expected type pd.Series or list, received - {type(x)}")

        if isinstance(x, list):
            x = pd.Series(x)

        temp = self.cv.transform(x)
        form_label = pd.DataFrame(temp.toarray()).apply(lambda _x: self.__classify_token_to_form(_x[_x > 0].index),
                                                        axis=1)
        return form_label

    # Convert token index to label
    def get_form_labels(self, form_index=None):
        if form_index is None:
            form_index = self.templates_.keys()

        return [(e, {v: k for k, v in self.cv.vocabulary.items()}[e]) for e in form_index]

    # Manually remove unwanted templates
    def remove_template(self, templates_to_remove):
        """
        Remove templates from the template instance
        :param templates_to_remove: Iterable object; Key(s) of of templates from TemplateClassifier.templates_
                                    that should be removed
        :return: None
        """
        if not hasattr(templates_to_remove, '__iter__'):
            raise TypeError(f"Expected 'templates_to_remove' to be iterable, but passed object of "
                            f"type - {type(templates_to_remove)}")
        new_templates = dict()
        for key, value in self.templates_.items():
            if key not in templates_to_remove:
                new_templates[key] = value
        self.templates_ = new_templates
        return

    # Load TemplateClassifier Model
    def save_model(self, object_name, level=1):
        """
        Dump TemplateClassifier instance for future use
        :param object_name: Path/Name of the file in which Model will be dumped
        :param level:   0: Only CountVectorizer Model & Templates
                        1: CountVectorizer Model & Templates
                            + Form Clusters
        :return: None
        """
        if level == 0:
            dump({'cv': self.cv,
                  'templates_': self.templates_,
                  'MIN_TOKENS': self.MIN_TOKENS}, open(object_name, 'wb+'))
        elif level == 1:
            dump({'cv': self.cv,
                  'templates_': self.templates_,
                  'km': self.km,
                  'MIN_TOKENS': self.MIN_TOKENS}, open(object_name, 'wb+'))
        else:
            raise ValueError("Invalid Level given")

    # Load TemplateClassifier Model
    def load_model(self, object_name, level=1):
        """
        Load TemplateClassifier instance for future use
        :param object_name: Path/Name of the file from which Model will be loaded
        :param level:   0: Only CountVectorizer Model & Templates
                        1: CountVectorizer Model & Templates
                            + Form Clusters
                        2: All
        :return: None
        """
        cache = load(open(object_name, 'rb'))
        if level == 0:
            self.cv = cache.get('cv', None)
            self.templates_ = cache.get('templates_', None)
            self.MIN_TOKENS = cache.get('MIN_TOKENS', None)
        elif level == 1:
            self.cv = cache.get('cv', None)
            self.templates_ = cache.get('templates_', None)
            self.MIN_TOKENS = cache.get('MIN_TOKENS', None)
            self.km = cache.get('km', None)
        else:
            raise ValueError("Invalid Level given")
