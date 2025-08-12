from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import resample
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
categories = ['rec.autos', 'sci.space', 'comp.graphics', 'talk.politics.misc']
data = fetch_20newsgroups(subset="all", categories=categories, remove=('headers', 'footers', 'quotes'))

x = data.data
y = data.target

df = pd.DataFrame({'text': x, 'target': y})

dataClass0 = df[df['target'] == 0]
dataClass1 = df[df['target'] != 0]

newData = resample(dataClass0, random_state=42, n_samples=100, replace=False)
df_new = pd.concat([newData, dataClass1])

x = df_new['text']
y = df_new['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vector = TfidfVectorizer(stop_words='english', max_df=0.9)
x_vect = vector.fit_transform(x_train)
x_pred_test = vector.transform(x_test)

model = LogisticRegression(class_weight='balanced',max_iter=1000)
model.fit(x_vect, y_train)
model1 = ComplementNB()
model1.fit(x_vect, y_train)
y_nb = model1.predict(x_pred_test)
y_pred = model.predict(x_pred_test)


# Step 11: Evaluation
print("Classification Report of Logistic Regression :\n")
print(classification_report(y_test, y_pred, target_names=categories, zero_division=0))
print("Classification Report of Naive Bayes: :\n")
print(classification_report(y_test, y_nb, target_names=categories, zero_division=0))

# Step 12: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(xticks_rotation=45, cmap='Reds')
plt.title("Confusion Matrix of Logistic Regression")
plt.tight_layout()
plt.show()

cm1 = confusion_matrix(y_test, y_nb)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=categories)
disp1.plot(xticks_rotation=45, cmap='Blues')
plt1.title("Confusion Matrix of Naive Bayes")
plt1.tight_layout()
plt1.show()