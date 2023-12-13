import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Normalize shapes of elements in 'data'
max_length = max(len(entry) for entry in data_dict['data'])
data = [entry + [0] * (max_length - len(entry)) for entry in data_dict['data']]

# Convert 'labels' to a NumPy array
labels = np.asarray(data_dict['labels'])

# Continue with the rest of your code
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)


