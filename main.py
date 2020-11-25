#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)
print(twenty_train.target_names, '\n', len(twenty_train.data), len(twenty_train.filenames))
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print("target names:", twenty_train.target_names[twenty_train.target[0]])
print("target:", twenty_train.target[:10])

#It's possible to get back the category names as follows:
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

#In order to perform ML on text documents, we first need to turn the text content into numerical feature vectors

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print("X_train_counts shape: ", X_train_counts.shape)

#CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted,
# the vectorizer has built a dictionary of feature indices:
print("Index value of world algorithm: ", count_vect.vocabulary_.get(u'algorithm'))

#The index value of a word in the vocabulary is linked to its frequency in the whole training corpus.

#Both tf and tf–idf can be computed as follows using TfidfTransformer:
#firstly use the fit(..) method to fit our estimator to the data and secondly the transform(..) method to transform our count-matrix to a tf-idf representation.
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print("X_train_tf shape: ", X_train_tf.shape)

#hese two steps can be combined to achieve the same end result faster by skipping redundant processing.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("X_train_tfidf shape: ", X_train_tfidf.shape)

# Train a classifier to predict the category of a post
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

#The %s specifier converts the object using str(), and %r converts it using repr().
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

#Build a pipeline to make the vectorizer => transformer => classifier easier to work with
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(twenty_train.data, twenty_train.target)

#Evaluation of the performance on the test set
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print("Predictive Accuracy（MultinomialNB）: ", np.mean(predicted == twenty_test.target))

#SVM: 1 of the best text classification algorithms
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(docs_test)
print("Predictive Accuracy（SVM）: ", np.mean(predicted == twenty_test.target))

# more detailed performance analysis of the results:
print("SVM classification report\n", metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))

print("SVM confusion matrix: \n", metrics.confusion_matrix(twenty_test.target, predicted))


#Parameter tuning using grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

print(gs_clf.predict(['God is love']))
print(gs_clf.predict(['God is love'])[0])
print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

#The object’s best_score_ and best_params_ attributes store the best mean score and the parameters setting corresponding to that score:
print("best score: ", gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %s" % (param_name, gs_clf.best_params_[param_name]))