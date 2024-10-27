import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

train_path = 'train.csv'
test_path = 'test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X_train_text = train_data['text']
X_train_additional = train_data[['sad', 'anger', 'surprise', 'disgust', 'fear', 'sentiment']]

X_test_text = test_data['text']
X_test_additional = test_data[['sad', 'anger', 'surprise', 'disgust', 'fear', 'sentiment']]

tfidf = TfidfVectorizer(max_features=500)
X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
X_test_tfidf = tfidf.transform(X_test_text).toarray()

X_train_combined = np.hstack((X_train_tfidf, X_train_additional))
X_test_combined = np.hstack((X_test_tfidf, X_test_additional))

emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']

results = {}

for emotion in emotions:
    y_train_emotion = train_data[emotion]
    y_test_emotion = test_data[emotion]
    
    model_emotion = LinearRegression()
    model_emotion.fit(X_train_combined, y_train_emotion)
    
    y_pred_emotion = model_emotion.predict(X_test_combined)
    
    mse = mean_squared_error(y_test_emotion, y_pred_emotion)
    r2 = r2_score(y_test_emotion, y_pred_emotion)
    
    results[emotion] = {'MSE': mse, 'R2': r2, 'y_test': y_test_emotion, 'y_pred': y_pred_emotion}

plt.figure(figsize=(18, 12))
for i, emotion in enumerate(emotions, 1):
    plt.subplot(2, 3, i)
    plt.scatter(results[emotion]['y_test'], results[emotion]['y_pred'], alpha=0.5, label='Predicted')
    plt.plot([min(results[emotion]['y_test']), max(results[emotion]['y_test'])],
             [min(results[emotion]['y_test']), max(results[emotion]['y_test'])], 'r--', label='Ideal Fit')
    plt.xlabel(f"Actual {emotion.capitalize()} Score")
    plt.ylabel(f"Predicted {emotion.capitalize()} Score")
    plt.title(f"Actual vs Predicted {emotion.capitalize()} Emotion Score")
    plt.legend()

plt.tight_layout()
plt.show()

for emotion, metrics in results.items():
    print(f"{emotion.capitalize()} - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}")
