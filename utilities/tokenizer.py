import re

from nltk import corpus

STOPWORDS = list(set(corpus.stopwords.words('english')))


def format_tokens(file_content, _remove='special_char', remove_stopwords=True):
    _remove = _remove.lower()
    if _remove not in {'alpha', 'digit', 'special_char'}:
        _remove = 'special_char'
    if type(file_content) == list:
        file_content = '\n'.join(file_content)
    if _remove == 'alpha':
        file_content = re.sub("[^ 0-9]", ' ', file_content)
    elif _remove == 'digit':
        file_content = re.sub("[^ a-zA-Z]", ' ', file_content)
    elif _remove == 'special_char':
        file_content = re.sub("[^ a-zA-Z0-9]", ' ', file_content)
    else:
        pass
    if remove_stopwords:
        file_content = ' '.join([word for word in file_content.split() if word not in STOPWORDS])
    file_content = ' '.join(file_content.split()).lower()
    return file_content
