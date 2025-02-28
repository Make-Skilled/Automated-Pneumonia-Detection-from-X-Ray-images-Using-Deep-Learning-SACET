from flask import Flask, request, render_template, session, redirect, url_for
import os
import sqlite3
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database Setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# Load Model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess image and make predictions
def model_predict(img_path, model):
    print(f"Processing Image: {img_path}")
    img = image.load_img(img_path, target_size=(150, 150), color_mode="grayscale")
    img = np.array(img)
    img = img.reshape(-1, 150, 150, 1)
    preds = model.predict(img)
    preds = np.argmax(preds, axis=-1)
    print(f"Prediction Output: {preds}")
    if preds[0] == 0:
        return "Severe Pneumonia"
    elif preds[0] == 1:
        return "Mild Pneumonia"
    else:
        return "Normal"

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            prediction = "No file selected"
        else:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            prediction = model_predict(file_path, model)
    
    return render_template('dashboard.html', user=session['user'], prediction=prediction)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user'] = username
            return redirect(url_for('dashboard'))
        return 'Invalid credentials'
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return 'Username already exists'
        conn.close()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

@app.route('/food_recommendation')
def food_recommendation():
    return render_template('food_recommendation.html')

@app.route('/remedies')
def remedies():
    return render_template('remedies.html')

@app.route('/recommendations/<stage>')
def recommendations(stage):
    if stage == "Severe Pneumonia":
        treatment = "Hospitalization and oxygen therapy recommended. Antibiotics required."
        food = "Nutritious diet with plenty of fluids, vitamin C, and protein."
        remedies = "Rest, steam inhalation, and prescribed medications."
    elif stage == "Mild Pneumonia":
        treatment = "Oral antibiotics and home care recommended. Monitor symptoms closely."
        food = "Hydration, warm soups, and foods rich in antioxidants."
        remedies = "Stay hydrated, avoid cold air, and take rest."
    else:
        treatment = "No treatment required. Maintain a healthy lifestyle."
        food = "Balanced diet with vitamins and minerals."
        remedies = "Regular exercise and proper hydration."
    return render_template('recommendations.html', stage=stage, treatment=treatment, food=food, remedies=remedies)

if __name__ == '__main__':
    app.run(debug=True)
