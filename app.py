from flask import Flask, render_template, request, redirect, url_for, session
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = load_model('model/kidney_disease_efficientnetv2b0.h5')
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Temporarily store registered users (username -> dict with email and password)
USERS = {}



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm']

        if username in USERS:
            return render_template('register.html', error="Username already exists")
        if password != confirm:
            return render_template('register.html', error="Passwords do not match")

        # Save user temporarily in memory
        USERS[username] = {
            'email': email,
            'password': password
        }

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = USERS.get(username)
        if user and user['password'] == password:
            session['user'] = username
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['file']
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            # Preprocess
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            pred_class = class_names[np.argmax(pred)]
            confidence = np.max(pred) * 100

            return render_template('result.html', filename=file.filename,
                                   prediction=pred_class,
                                   confidence=confidence)
    return render_template('upload.html')

@app.route('/charts')
def charts():
    labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
    values = [200, 400, 300, 280]  # Example

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.title("Sample Class Distribution")

    chart_path = os.path.join('static', 'chart.png')
    plt.savefig(chart_path)
    plt.close()  # IMPORTANT: Avoids memory leaks and tkinter errors

    return render_template('charts.html', chart_url=chart_path)


if __name__ == '__main__':
    app.run(debug=True)
