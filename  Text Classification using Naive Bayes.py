#Text Classification using Naive Bayes
# Experiment 1: Text Classification using Naive Bayes (20 Newsgroups)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample

# There are completely 20 categories in the 20 Newsgroups dataset. and i want to compare only the 4 categories that i have mentioned
categories = ['rec.autos', 'sci.space', 'comp.graphics', 'talk.politics.misc']

# I am loading only the categories which i want and i am not loading all the reamining 16 of them
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Step 3: Separate features and labels
# This contains the text data like a paragraph or a sentence
X = data.data
# This contains the target value like 0,1,2,3
y = data.target

# Step 4: Create DataFrame for balancing
# This creates a table like structure by keeping text and its target value to it
# This is a political news 1, This is a news related to cars 2 something like that
df = pd.DataFrame({'text': X, 'target': y})

# In the data the group with target value 0 is the largest group so we are taking the larger group into one and the rest of the groups into another
df_class0 = df[df['target'] == 0]           # Class 0 (rec.autos)
# Here all the remaining groups are combined
df_others = df[df['target'] != 0]           # Other classes

# Downsample class 0 to 100 samples
# Here the total large smaple of target 0 is reduced to 100 samples
df_class0_downsampled = resample(df_class0, replace=False, n_samples=100, random_state=42)

# Combine balanced data
# we are combining the new 100 samples of target 0 with the rest of the groups
df_balanced = pd.concat([df_class0_downsampled, df_others])

# Step 6: Update features and labels
# Here this stores the text like "This car is mine"
X = df_balanced['text']
# Here this stores the value of the text like 0,1,2,3
y = df_balanced['target']

# Step 7: Split into train and test sets
# It splits the data into training set and testing set, with 80% for training and 20% for testing. Randon state 42 makes the set to be same for evey run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: TF-IDF vectorization
# Removes words like the,an,a and all that. max_df 0.9 means that if a word appears in more than 90% of the documents, it will be ignored
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
# This ccalculates the weightage of the words like car, nasa, mars will high weightage in considering them related to the category
X_train_vec = vectorizer.fit_transform(X_train)
# This will do the same but for the test data in the same algorithm used to convert the train data into vector(number)
X_test_vec = vectorizer.transform(X_test)

# Step 9: Train Complement Naive Bayes model
# This will create a model of the naive bayes
model = ComplementNB()
# This will train the model with the train data and its target value
model.fit(X_train_vec, y_train)

# Step 10: Predict
# This will take the test data and predict the values for the test data and store it in y_pred
y_pred = model.predict(X_test_vec)

# Step 11: Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=categories, zero_division=0))

# Step 12: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(xticks_rotation=45, cmap='Reds')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()