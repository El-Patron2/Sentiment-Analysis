import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load the dataset
data = pd.read_csv('electronics.csv')

# Preprocess the data
stop_words = set(stopwords.words('english'))
data['normalized_review_body'] = data['normalized_review_body'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in stop_words and word not in string.punctuation]) if isinstance(x, str) else '')

# Split the data into features and labels
X = data['normalized_review_body']
y = data['review_body_sentiment_label']

# Remove rows with empty strings
mask = X != ''
X = X[mask]
y = y[mask]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Scale the data without centering
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000, solver='liblinear', verbose=1)
model.fit(X_scaled, y)

# Create an instance of the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# User registration and authentication
users = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if the username or email already exists
        if username in users or email in [user['email'] for user in users.values()]:
            flash('Username or email already exists', 'danger')
            return redirect(url_for('register'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Store the user information
        users[username] = {'email': email, 'password': hashed_password}

        flash('Registration successful', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username exists and the password is correct
        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            flash('Login successful', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash('Please log in to access this page', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        review_text = request.form['review_text']

        # Preprocess the review text
        review_text = ' '.join([word for word in review_text.split() if word.lower() not in stop_words and word not in string.punctuation])

        # Transform the review text using the TF-IDF vectorizer
        review_vector = vectorizer.transform([review_text])

        # Scale the review vector
        review_vector_scaled = scaler.transform(review_vector)

        # Make the prediction using the logistic regression model
        prediction = model.predict(review_vector_scaled)[0]

        # Predict the sentiment using the SentimentIntensityAnalyzer
        scores = sid.polarity_scores(review_text)
        compound_score = scores['compound']

        if compound_score >= 0.05:
            sentiment = 'Positive'
            emoji = '🙂'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
            emoji = '🙁'
        else:
            sentiment = 'Neutral'
            emoji = '😐'

        # Filter the data DataFrame based on the predicted sentiment
        filtered_products = data[data['review_body_sentiment_label'] == prediction]

        return render_template('predict.html', sentiment=sentiment, emoji=emoji, products=filtered_products)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)