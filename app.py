import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import mysql.connector
from werkzeug.utils import secure_filename
from getfeatures import get_features  
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


app = Flask(__name__)
app.secret_key = "JD_KingOfMonsters"
UPLOAD_FOLDER = r"C:\Users\tonyw\Desktop\Projects\PY\GallerySearch\images"  
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "heic"}


DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",  
    "database": "gsearch"
}

def connect_to_db():
    """Establish connection to the MySQL database."""
    return mysql.connector.connect(**DB_CONFIG)

def check_duplicate_in_db(image_name):
    """Check if an image name already exists in the database."""
    try:
        db = connect_to_db()
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM object_details WHERE image_name = %s", (image_name,))
        count = cursor.fetchone()[0]
        cursor.close()
        db.close()
        return count > 0
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        flash(f"Database error: {err}")
        return False

def store_features_in_db(image_name, set_labels, captions):
    """Store extracted features and captions into MySQL database."""
    try:
        db = connect_to_db()
        cursor = db.cursor()
        cursor.execute(
            "REPLACE INTO object_details (image_name, features, captions) VALUES (%s, %s, %s)",
            (image_name, set_labels, captions),
        )
        db.commit()
        cursor.close()
        db.close()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        flash(f"Database error: {err}")

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
        flash(f"Database error: {err}")

def handle_duplicate_filename(filename):
    """Handle duplicate filenames by appending a counter."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

def extract_features(image_path):
    """Extract object details and captions from the image using get_features."""
    features_data = get_features(image_path)
    set_labels = ", ".join(features_data['set_labels'])  
    captions = features_data['captions']  
    return set_labels, captions

def allowed_file(filename):
    """Check if the uploaded file is an allowed image type."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def count_image_formats():
    """Counts the number of PNG, JPG, and JPEG images in the database."""
    try:
        
        connection = connect_to_db()
        cursor = connection.cursor()

        
        query = """
            SELECT
                SUM(CASE WHEN image_name LIKE '%.png' THEN 1 ELSE 0 END) AS png_count,
                SUM(CASE WHEN image_name LIKE '%.jpg' THEN 1 ELSE 0 END) AS jpg_count,
                SUM(CASE WHEN image_name LIKE '%.jpeg' THEN 1 ELSE 0 END) AS jpeg_count,
                SUM(CASE WHEN image_name LIKE '%.heic' THEN 1 ELSE 0 END) AS heic_count
            FROM object_details;
        """
        
        cursor.execute(query)
        cnt_result = cursor.fetchone()
        
    
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        
        if connection.is_connected():
            cursor.close()
            connection.close()
    
    return cnt_result

@app.route("/")
def gallery():
    """Display the gallery with all images."""
    images = os.listdir(app.config["UPLOAD_FOLDER"])
    
    cnt_res = count_image_formats()
    return render_template("gallery.html", images=images, cnt_res=cnt_res)

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

            
            if check_duplicate_in_db(filename):
                flash(f"Duplicate filename '{filename}' detected. Skipping upload.")
                continue

            filename = handle_duplicate_filename(filename)  
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            set_labels, captions = extract_features(filepath)
            store_features_in_db(filename, set_labels, captions)

    flash("Images uploaded successfully!")
    return redirect(url_for("gallery"))

@app.route("/delete/<filename>", methods=["POST"])
def delete_image(filename):
    """Delete an image from the gallery."""
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(filepath):
        os.remove(filepath)  
        delete_features_from_db(filename)  
        flash("Image deleted successfully!")
    else:
        flash("Image not found.")

    return redirect(url_for("gallery"))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

def fetch_data_from_db():
    """Fetch image names, features, and captions from the database."""
    try:
        db = connect_to_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT image_name, features, captions FROM object_details")
        records = cursor.fetchall()
        cursor.close()
        db.close()
        return records
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return []

@app.route("/search", methods=["GET"])
def search():
    """Search images by captions."""
    search_query = request.args.get("query")
    cnt_res = count_image_formats()
    rec = []
    if search_query:
        records = fetch_data_from_db()
        cnt_res = count_image_formats()
        image_names = [record['image_name'] for record in records]
        features = [record['features'] for record in records]
        captions = [record['captions'] for record in records]

        
        combined_texts = [f"{features[i]} {captions[i]}" for i in range(len(records))]
        vectorizer = TfidfVectorizer(stop_words='english')
        all_texts = [search_query] + combined_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        similarities = cosine_sim.flatten()
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_image_info = [(image_names[i], f"{UPLOAD_FOLDER}\\{image_names[i]}") for i in sorted_indices]
        final_lst = []
        for result in sorted_image_info:
            final_lst.append(result[0])
        print(final_lst)

        return render_template("gallery.html", images = final_lst, cnt_res=cnt_res)
    return redirect(url_for("gallery"))

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)  
    app.run(debug=True)
