from template import TemplateClassifier


tc = TemplateClassifier()
print(tc.vocabulary_)

tc.fit_for_template(['cc'], ['c'], '')

tc.