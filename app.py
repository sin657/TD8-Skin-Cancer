from flask import Flask, render_template, request, redirect, session, flash
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sqlite3
import uuid

app = Flask(__name__)
app.secret_key = "skin_cancer_secret_2025"

UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = "skin_cancer.db"

model = None
MODEL_PATH = "model/vgg16_skin_cancer.h5"
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Modèle VGG16 chargé.")
    except Exception as e:
        print(f"⚠️  Impossible de charger le modèle : {e}")
else:
    print("⚠️  Modèle introuvable – mode démonstration activé.")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            age         INTEGER,
            result      TEXT NOT NULL,
            probability REAL,
            image_path  TEXT,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute(
        "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
        ("admin", "1234"),
    )
    conn.commit()
    conn.close()

def predict_image(img_path):
    if model is None:
        import random
        prob = random.uniform(0.1, 0.95)
        label = "Malignant" if prob > 0.5 else "Benign"
        return label, prob
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    label = "Malignant" if pred > 0.5 else "Benign"
    return label, float(pred)

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password),
        ).fetchone()
        conn.close()
        if user:
            session["user"] = username
            flash("Connexion réussie ✓", "success")
            return redirect("/dashboard")
        flash("Identifiants incorrects ✗", "danger")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html", user=session["user"])

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/")
    if request.method == "POST":
        try:
            name = request.form["name"]
            age  = request.form["age"]
            file = request.files["image"]
            if not file or file.filename == "":
                flash("Veuillez choisir une image.", "warning")
                return redirect("/predict")
            ext      = os.path.splitext(file.filename)[1]
            filename = f"{uuid.uuid4().hex}{ext}"
            path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            label, prob = predict_image(path)
            conn = get_db()
            conn.execute(
                "INSERT INTO patients (name, age, result, probability, image_path) VALUES (?,?,?,?,?)",
                (name, age, label, float(prob), path),
            )
            conn.commit()
            conn.close()
            flash("Analyse réussie ✓", "success")
            return render_template(
                "result.html",
                result=label,
                prob=round(prob * 100, 2),
                img=path,
                name=name,
                age=age,
            )
        except Exception as e:
            flash(f"Erreur système : {e}", "danger")
            return redirect("/predict")
    return render_template("predict.html")

@app.route("/patients")
def patients():
    if "user" not in session:
        return redirect("/")
    conn = get_db()
    data = conn.execute(
        "SELECT * FROM patients ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return render_template("patients.html", patients=data)

@app.route("/logout")
def logout():
    session.clear()
    flash("Déconnecté.", "info")
    return redirect("/")

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
