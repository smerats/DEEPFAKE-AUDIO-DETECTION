from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import sqlite3
from werkzeug.utils import secure_filename

# ----------------- Flask App -----------------
app = Flask(__name__)
app.secret_key = "audio_guard_secret"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- DATABASE -----------------
DB_NAME = "audio_results.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        prediction TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ----------------- CNN MODEL -----------------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1,16,3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16*19,32)
        self.fc2 = nn.Linear(32,2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load model
model = CNNModel()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

print("Model Loaded Successfully!")

# ----------------- PREDICTION FUNCTION -----------------
def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(mfcc_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    detected = "real" if pred_idx.item() == 0 else "fake"
    confidence = round(confidence.item()*100,2)

    return detected, confidence

# ----------------- ROUTES -----------------

# Login
@app.route('/', methods=['GET','POST'])
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('upload'))
    return render_template('login.html')

# Signup
@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('signup.html')

# Upload
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':

        if 'audioFile' not in request.files:
            return "No file part"

        file = request.files['audioFile']

        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 🔥 PREDICT
            detected, confidence = predict_audio(filepath)

            # 🔥 SAVE TO DATABASE
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO results (filename, prediction, confidence) VALUES (?, ?, ?)",
                (filename, detected, confidence)
            )

            conn.commit()
            conn.close()

            return redirect(url_for('result', detected=detected, confidence=confidence))

    return render_template('upload.html')

# Record page
@app.route('/record')
def record():
    return render_template('record.html')

# Record audio
@app.route('/record_audio', methods=['POST'])
def record_audio():
    if 'audioBlob' not in request.files:
        return redirect(url_for('upload'))

    file = request.files['audioBlob']

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        detected, confidence = predict_audio(filepath)

        # 🔥 SAVE TO DATABASE
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO results (filename, prediction, confidence) VALUES (?, ?, ?)",
            (filename, detected, confidence)
        )

        conn.commit()
        conn.close()

        return redirect(url_for('result', detected=detected, confidence=confidence))

    return redirect(url_for('upload'))

# Result page
@app.route('/result')
def result():
    detected = request.args.get('detected', 'real')
    confidence = request.args.get('confidence', 0)
    return render_template('result.html', detected=detected, confidence=confidence)

# 🔥 HISTORY PAGE
@app.route('/history')
def history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM results")
    data = cursor.fetchall()

    conn.close()

    return render_template('history.html', data=data)

# ----------------- RUN -----------------
if __name__ == '__main__':
    app.run(debug=True)