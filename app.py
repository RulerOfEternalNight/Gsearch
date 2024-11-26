import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import cv2
import mysql.connector
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array

# Configuration
app = Flask(__name__)
app.secret_key = "JD_KingOfMonsters"
UPLOAD_FOLDER = r"C:\Users\tonyw\Desktop\Projects\PY\GallerySearch\images"  # Local folder to store images
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# MySQL configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "gsearch"
}

# Load pre-trained model
model = MobileNetV2(weights="imagenet")


def connect_to_db():
    """Establish connection to the MySQL database."""
    return mysql.connector.connect(**DB_CONFIG)


def store_features_in_db(image_name, features):
    """Store extracted features into MySQL database."""
    try:
        db = connect_to_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO object_details (Features, image_name) VALUES (%s, %s)",
            (features, image_name),
        )
        db.commit()
        cursor.close()
        db.close()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")


def delete_features_from_db(image_name):
    """Delete image details from the database."""
    try:
        db = connect_to_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM object_details WHERE image_name = %s", (image_name,))
        db.commit()
        cursor.close()
        db.close()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")


def extract_features(image_path):
    """Extract object details from the image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = image.reshape((1, *image.shape))

    predictions = model.predict(image)
    decoded = decode_predictions(predictions, top=3)[0]
    features = ", ".join([label for (_, label, _) in decoded])  # Extract only labels
    return features


def allowed_file(filename):
    """Check if the uploaded file is an allowed image type."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def gallery():
    """Display the gallery with all images."""
    images = os.listdir(app.config["UPLOAD_FOLDER"])
    return render_template("gallery.html", images=images)


@app.route("/upload", methods=["POST"])
def upload_image():
    """Handle image upload."""
    if "files[]" not in request.files:
        flash("No files part")
        return redirect(request.url)

    files = request.files.getlist("files[]")
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Extract features and store in DB
            features = extract_features(filepath)
            store_features_in_db(filename, features)

    flash("Images uploaded successfully!")
    return redirect(url_for("gallery"))


@app.route("/delete/<filename>", methods=["POST"])
def delete_image(filename):
    """Delete an image from the gallery."""
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(filepath):
        os.remove(filepath)  # Remove from local folder
        delete_features_from_db(filename)  # Remove from database
        flash("Image deleted successfully!")
    else:
        flash("Image not found.")

    return redirect(url_for("gallery"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)  # Create folder if it doesn't exist
    app.run(debug=True)
