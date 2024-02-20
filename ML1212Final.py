# %%
import json
import os
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

parent_dir = os.getcwd()
stop_words=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# %%
with open(os.path.join(parent_dir,'dataset/train.json'),'r') as k:
    data = json.load(k)
    for row in data:
        abstract = row['abstract']
        tokens = word_tokenize(abstract)
        tokens_without_stopwords = [t for t in tokens if t not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens_without_stopwords]
        row['abstract_processed'] = lemmatized_tokens
    df = pd.DataFrame(data) 

# %%

# create an instance of the MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# fit the MultiLabelBinarizer to the clean abstract column
mlb.fit_transform(df['abstract_processed'])

# transform the clean abstract column
mlb_transformed = mlb.fit_transform(df['abstract_processed'])

# get the names of the labels
mlb_labels = mlb.classes_

# add the binarized labels as columns to the data frame
df_binarized = pd.concat([df,pd.DataFrame(mlb_transformed, columns=mlb_labels)], axis=1)

# print the data frame
df_binarized.head()

# %%
X = df_binarized.drop(columns=['paperId', 'authorName','venue','authorId', 'title', 'abstract','abstract_processed','year'])
y = df_binarized['authorId']

# %%

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.14, random_state = 43)

# Train the model using LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
# Evaluate the model
accuracy = model.score(X_test, y_test)
print('Model accuracy: ', accuracy)


# %%
model.fit(X,y)


# %%

with open(os.path.join(parent_dir,'dataset/test.json'),'r') as k:
    data = json.load(k)
    for row in data:
        abstract = row['abstract']
        tokens = word_tokenize(abstract)
        tokens_without_stopwords = [t for t in tokens if t not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens_without_stopwords]
        row['abstract_processed'] = lemmatized_tokens
    df_test = pd.DataFrame(data) 

df_test.head()

# %%

# fit the MultiLabelBinarizer to the clean abstract column
mlb_transformed = mlb.transform(df_test['abstract_processed'])

# get the names of the labels
mlb_labels = mlb.classes_

# add the binarized labels as columns to the data frame
df_binarized_test = pd.concat([df_test,pd.DataFrame(mlb_transformed, columns=mlb_labels)], axis=1)

X_jasontest = df_binarized_test.drop(columns=['paperId','venue','title', 'abstract','abstract_processed','year'])

# %%

y_pred = model.predict(X_jasontest)

X_jasontest.head()

# %%

df_test['authorId'] = y_pred.tolist()
pred = df_test[['paperId', 'authorId']]
pred = pred.to_dict('records')

with open('results/predictions.json', 'w', encoding='utf-8') as f:
    json.dump(pred,f,indent=4)
