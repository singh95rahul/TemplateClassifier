from itertools import chain
from pickle import load, dump

import pandas as pd
from numpy import NaN, min
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

            data_trans_df = self.__fit_cv(data, **kwargs)

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
    def predict_cluster(self):
        raise NotImplementedError()

    # Train TemplateClassifier Model for template types - Using Tokens
    def fit_for_template(self, data, template_headers, tokens_p_template, template_sep=None, **kwargs):
        """
        Fit the TemplateClassifier with data to extract the common templates' tokens. This method will fit the TC
        and generate token that can be used to identify different common templates among the data.
        :param data: Pd.Series or List; Dataset than should be fitted to the CV
        :param template_headers: List; Sample templates token. Like Form Name etc.
        :param tokens_p_template: int; Number of tokens to be extracted from each template
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

        if isinstance(data, list):
            data = pd.Series(data)

        print(" |+ Analyzing Contents +|")
        data_transformed = self.__fit_cv(data, **kwargs)
        data_trans_df = pd.DataFrame(data_transformed.toarray(), columns=self.cv.get_feature_names())

        print(" |+ Extracting templates to fit..", end='')
        # Extracting similar template headers
        template_headers = list(chain(
            *[[feature for feature in self.cv.get_feature_names() if header in feature] for header in
              template_headers]))

        # Extracting additional tokens which are part of, templates that has above headers
        # Filtering contents that has template headers
        data_trans_df = data_trans_df.loc[data_trans_df[template_headers].apply(lambda x: any(x > 0),
                                                                                axis=1)].replace(0, NaN)
        data = data.iloc[data_trans_df.dropna(thresh=self.MIN_TOKENS, axis=1).index]
        print(" Done +|")

        # Separating content into individual templates to similar templates
        if template_sep is not None:
            print(f" |+ Splitting Templates form content, using separator - '{template_sep}'.. ", end='')
            data = data.str.split(template_sep.lower())
            data = data.apply(pd.Series).stack().reset_index()[0].replace('', NaN).dropna().reset_index(drop=True)
            print(" Done +|")

        data_trans_df = self.__fit_cv(data, **kwargs)

        # Filtering Templates that had some token
        print(" |+ Generating Standard templates.. ", end='')
        self.__form_template_data = data_trans_df[
            (data_trans_df.sum(axis=1) > int(self.MIN_TOKENS)).reshape(-1).nonzero()[-1]].toarray()
        self.__form_template_data = pd.DataFrame(self.__form_template_data)

        self.templates_ = dict()

        for form in [self.cv.vocabulary.get(e, None) for e in template_headers]:
            if form is not None:
                temp_df = self.__form_template_data.loc[self.__form_template_data[form] > 0].replace(0, NaN)
                temp_df = temp_df.dropna(thresh=self.MIN_TOKENS, axis=1)
                self.templates_[form] = set(
                    temp_df.count(axis=0).sort_values(ascending=False)[:tokens_p_template].index).union([form])

        print(" Done +|")
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

    # Load TemplateClassifier Model
    def save_model(self, object_name, level=0):
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
    def load_model(self, object_name, level=0):
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
