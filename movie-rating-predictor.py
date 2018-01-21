from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import string
from nltk.corpus import stopwords

def cleaner(review):
    nopunc = [char for char in review if char not in string.punctuation]
    polished = "".join(nopunc)
    polished = polished.lower()
    return " ".join([word for word in polished.split() if word not in stop])

stop = stopwords.words('english')
for i in range(len(stop)):
    stop[i] = str(stop[i])

train_data = pd.read_table('labeledTrainData.tsv')
test_data = pd.read_table('testData.tsv')

train_data['new_review'] = train_data['review'].apply(cleaner)
test_data['new_review'] = test_data['review'].apply(cleaner)

CV = CountVectorizer(ngram_range=(1,2))
BOW = CV.fit_transform(train_data['new_review']) #implicitly fitted CV

TFIDF = TfidfTransformer()
TFIDF.fit(BOW)
X_train = TFIDF.transform(BOW)
y_train = train_data['sentiment']

X_test = CV.transform(test_data['new_review'])
X_test = TFIDF.transform(X_test)

model = MultinomialNB()

model.fit(X_train, y_train)
prediction = model.predict(X_test)

test_data['sentiment'] = prediction
test_data.to_csv('movie-rating-predictions.csv')


