from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
import joblib

import pandas as pd
import argparse, os
import numpy as np

########### NLP #############
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords=stopwords.words('spanish')

################ visualization ################
import seaborn as sns
import matplotlib.pyplot as plt

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--random-state', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--model-dir', type=str, default='tmp')
#     parser.add_argument('--training', type=str, default='spanish_final.csv')
    
    args, _ = parser.parse_known_args()
    normalize = args.normalize
    test_size = args.test_size
    random_state = args.random_state
    model_dir  = args.model_dir
    training_dir = args.training
       
    filename = os.path.join(training_dir, 'spanish_final.csv')
    df = pd.read_csv(filename)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    df['category_id'] = df['Etiquetas'].factorize()[0]
    
    colslist = ['Etiquetas','Mensaje','category_id']
    df.columns = colslist
    df.head(5)
    Index=df['Etiquetas'].value_counts()
    
    # Porter Stemming
    df['Mensaje_porter_stemmed'] = df['Mensaje'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    #df.head()
    
    #Converting TO LowerCase
    df['Mensaje_porter_stemmed'] = df['Mensaje_porter_stemmed'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    
    #Removing Punctuation
    df['Mensaje_porter_stemmed'] = df['Mensaje_porter_stemmed'].str.replace('[^\w\s]','')
    
    #Low frequency term filtering (count < 3)
    freq = pd.Series(' '.join(df['Mensaje_porter_stemmed']).split()).value_counts()
    freq2 = freq[freq <= 3]
    #freq2
    freq3 = list(freq2.index.values)
    #freq3
    df['Mensaje_porter_stemmed'] = df['Mensaje_porter_stemmed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (freq3)]))
    
    #TfidfVectorizer
    df = df[['Etiquetas', 'category_id', 'Mensaje_porter_stemmed']]

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.Mensaje_porter_stemmed).toarray()
    labels = df.category_id
    features.shape
    features
    
    df.columns = ['Etiquetas', 'category_id', 'Mensaje_porter_stemmed']
    category_id_df = df[['Etiquetas', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Etiquetas']].values)
    
    #chi2
    N = 3
    for newstype, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(newstype))
        print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
        
    print(labels)
    print('before apply over sampling methord: ',df['Etiquetas'].value_counts())  
    #KFold validation
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    
    
    
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    
    print(X_train.shape)
    print(y_train.shape)
    # handle over sampling
#     smote = SMOTE()
#     X_Train , y_Train = smote.fit_resample(X_train , y_train)
#     print('After apply over sampling methord: ',df['labels'].value_counts())
  
    
    model.fit(X_train, y_train)


    regr = LogisticRegression(random_state = random_state)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    score =accuracy_score(y_test,y_pred)
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    print('Test Accuracy Score', score)
    print('Predicted Result: ',y_pred)
    print('Actual Result: ',y_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    
    model = os.path.join(model_dir, 'model.joblib')
    joblib.dump(regr, model)