import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import mysql.connector
from werkzeug.utils import secure_filename
from getfeatures import get_features  # Ensure getfeatures.py is in the same directory
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
    "password": "",  # Update with your MySQL password
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
    set_labels = ", ".join(features_data['set_labels'])  # Convert set_labels to comma-separated string
    captions = features_data['captions']  # Get captions as is
    return set_labels, captions

def allowed_file(filename):
    """Check if the uploaded file is an allowed image type."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def count_image_formats():
    """Counts the number of PNG, JPG, and JPEG images in the database."""
    try:
        # Connect to the database
        connection = connect_to_db()
        cursor = connection.cursor()

        # Query to count image formats
        query = """
            SELECT
                SUM(CASE WHEN image_name LIKE '%.png' THEN 1 ELSE 0 END) AS png_count,
                SUM(CASE WHEN image_name LIKE '%.jpg' THEN 1 ELSE 0 END) AS jpg_count,
                SUM(CASE WHEN image_name LIKE '%.jpeg' THEN 1 ELSE 0 END) AS jpeg_count
            FROM object_details;
        """
        
        cursor.execute(query)
        cnt_result = cursor.fetchone()
        # png_count, jpg_count, jpeg_count = result
    
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()
    
    return cnt_result

@app.route("/")
def gallery():
    """Display the gallery with all images."""
    images = os.listdir(app.config["UPLOAD_FOLDER"])
    # print(images)
    cnt_res = count_image_formats()
    # try:
    #     db = connect_to_db()
    #     cursor = db.cursor(dictionary=True)
    #     cursor.execute("SELECT image_name, captions FROM object_details")
    #     captions_data = cursor.fetchall()
    #     cursor.close()
    #     db.close()
    # except mysql.connector.Error as err:
    #     print(f"Database error: {err}")
    #     flash(f"Database error: {err}")
    #     captions_data = []

    # captions_dict = {item['image_name']: item['captions'] for item in captions_data}
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

            # Check for duplicate filename in the database
            if check_duplicate_in_db(filename):
                flash(f"Duplicate filename '{filename}' detected. Skipping upload.")
                continue

            filename = handle_duplicate_filename(filename)  # Handle duplicates locally
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

# @app.route("/search", methods=["GET"])
# def search():
#     """Search images by captions."""
#     search_query = request.args.get("query")
#     if search_query:
#         try:
#             db = connect_to_db()
#             cursor = db.cursor(dictionary=True)
#             cursor.execute("SELECT image_name, captions FROM object_details WHERE captions LIKE %s", (f"%{search_query}%",))
#             search_results = cursor.fetchall()
#             cursor.close()
#             db.close()
#         except mysql.connector.Error as err:
#             print(f"Database error: {err}")
#             flash(f"Database error: {err}")
#             search_results = []

#         search_results_dict = {item['image_name']: item['captions'] for item in search_results}
#         return render_template("gallery.html", images=[item['image_name'] for item in search_results], captions=search_results_dict)
#     return redirect(url_for("gallery"))

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

        # Combine features and captions into one list for comparison
        combined_texts = [f"{features[i]} {captions[i]}" for i in range(len(records))]

        # Initialize TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        # Combine the query with the database features and captions
        all_texts = [search_query] + combined_texts

        # Vectorize the query and the database entries
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Compute cosine similarity between the query and each entry in the database
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

        # Get the list of similarities with the query
        similarities = cosine_sim.flatten()

        # Sort the images based on similarity in descending order
        sorted_indices = np.argsort(similarities)[::-1]

        # Get the sorted image names by relevance
        sorted_image_info = [(image_names[i], f"{UPLOAD_FOLDER}\\{image_names[i]}") for i in sorted_indices]
        
        final_lst = []
        for result in sorted_image_info:
            final_lst.append(result[0])

        print(final_lst)

        # search_results_dict = {item['image_name']: item['captions'] for item in search_results}
        return render_template("gallery.html", images = final_lst, cnt_res=cnt_res)
    return redirect(url_for("gallery"))

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)  # Create folder if it doesn't exist
    app.run(debug=True)
