from flask import Flask, request, render_template, session, redirect, url_for, flash, send_from_directory
import os
import sqlite3
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database Setup
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Users table with hashed passwords
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
    
    # Scan history table
    cursor.execute('''CREATE TABLE IF NOT EXISTS scan_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

init_db()

# Load Model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

# Check if binary or multi-class model
if model.output_shape[-1] == 1:
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    is_binary = True
else:
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    is_binary = False

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Function to preprocess image and make predictions
def model_predict(img_path, model, username):
    print(f"Processing Image: {img_path}")

    # Load image and normalize
    img = image.load_img(img_path, target_size=(150, 150), color_mode="grayscale")
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(-1, 150, 150, 1)  # Ensure correct shape

    # Make prediction
    preds = model.predict(img)

    # Binary classification (Pneumonia or Normal)
    if is_binary:
        pneumonia_prob = 1 - preds[0][0]
        result = f"Pneumonia ({pneumonia_prob * 100:.2f}% severity)" if pneumonia_prob > 0.5 else "Normal"
    
    # Multi-class classification (Severe, Mild, Normal)
    else:
        class_idx = np.argmax(preds, axis=-1)[0]
        pneumonia_prob, mild_prob, normal_prob = preds[0] * 100
        result = ["Severe Pneumonia", "Mild Pneumonia", "Normal"][class_idx] + f" ({preds[0][class_idx] * 100:.2f}% severity)"

    # Store scan result in database
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO scan_history (username, image_path, prediction) VALUES (?, ?, ?)", 
                       (username, img_path, result))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error storing scan history: {e}")

    return result

@app.route('/')
def home():
    return redirect(url_for('dashboard')) if 'user' in session else redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash("No file selected. Please upload an image.", "danger")
        else:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = model_predict(file_path, model, session['user'])
    
    return render_template('dashboard.html', user=session['user'], prediction=prediction)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):  # Verify hashed password
            session['user'] = username
            return redirect(url_for('dashboard'))

        flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.', 'danger')
        finally:
            conn.close()
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, prediction, timestamp FROM scan_history WHERE username = ? ORDER BY timestamp DESC", 
                   (session['user'],))
    history_data = cursor.fetchall()
    conn.close()

    return render_template('history.html', history=history_data)

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
    data = {
        "Severe Pneumonia": {
            "treatment": "Hospitalization and oxygen therapy recommended. Antibiotics required.",
            "food": "Nutritious diet with plenty of fluids, vitamin C, and protein.",
            "remedies": "Rest, steam inhalation, and prescribed medications."
        },
        "Mild Pneumonia": {
            "treatment": "Oral antibiotics and home care recommended. Monitor symptoms closely.",
            "food": "Hydration, warm soups, and foods rich in antioxidants.",
            "remedies": "Stay hydrated, avoid cold air, and take rest."
        },
        "Normal": {
            "treatment": "No treatment required. Maintain a healthy lifestyle.",
            "food": "Balanced diet with vitamins and minerals.",
            "remedies": "Regular exercise and proper hydration."
        }
    }

    return render_template('recommendations.html', stage=stage, **data.get(stage, {}))

if __name__ == '__main__':
    app.run(debug=True)