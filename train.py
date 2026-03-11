import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from model import SpamClassifier

print("Loading data…")
df = pd.read_csv(
    'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
    sep='\t', header=None, names=['label', 'message']
)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer()
X = torch.tensor(vectorizer.fit_transform(df['message']).toarray(), dtype=torch.float32)
y = torch.tensor(df['label'].values, dtype=torch.float32)

X_train, _, y_train, _ = train_test_split(X, y, train_size=0.80, random_state=42)

print("Training…")
model = SpamClassifier(input_size=X_train.shape[1])
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train.reshape(-1, 1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model.pth')
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Done — saved model.pth and vectorizer.pkl")
