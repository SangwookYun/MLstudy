from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
tfidf = vec.fit_transform([
    'I like machine learning and clusetring algorithms',
    'Apples, oranges, and kind of fruits are healthy',
    'Is it feasible with machine learning algorithms?',
    'My family is happy because of the healthy fruits'
])

print((tfidf*tfidf.T).A)

