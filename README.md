# FotoFind: An Advanced Image Retrieval System

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Models and Methodology](#models-and-methodology)
- [Implementation Details](#implementation-details)
- [Development Environment](#development-environment-and-project-info)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

## Introduction
FotoFind is a robust and feature-rich image retrieval system designed to efficiently store, manage, and search large-scale image collections. By integrating object detection, image captioning, and Optical Character Recognition (OCR), FotoFind enhances metadata extraction and enables efficient text-based image search.

## Features
- **Automated Feature Extraction:** Uses advanced deep learning models to detect objects, generate captions, and extract text from images.
- **Searchable Image Metadata:** Stores object labels, captions, and OCR text in a structured MySQL database.
- **User-Centric Interface:** Flask-based web application for uploading, browsing, and searching images.
- **Efficient Search Mechanism:** Uses TF-IDF vectorization and cosine similarity to rank search results based on relevance.

## System Architecture
FotoFind follows a modular, scalable architecture comprising:
1. **Flask Web Server:** Manages user interactions and file uploads.
2. **Feature Extraction Pipeline:** Implements YOLO/Faster-RCNN (object detection), BLIP/ViT-GPT2 (captioning), and EasyOCR (text extraction).
3. **Metadata Database (MySQL):** Stores metadata for efficient retrieval.
4. **Local Storage:** Stores uploaded images.
5. **Search Mechanism:** Computes TF-IDF vectors and ranks results using cosine similarity.

## Models and Methodology

![](/res/fotofind_page-0004.jpg)

### Object Detection
- **Faster R-CNN:** Two-stage object detection framework for high accuracy.
- **YOLO:** Single-shot detector for real-time object recognition.

### Image Captioning
- **ViT-GPT2:** Vision Transformer with GPT-2 for coherent captions.
- **BLIP:** Bootstrapped Language-Image Pre-training for enhanced captioning accuracy.

### Optical Character Recognition (OCR)
- **EasyOCR:** Deep learning-based OCR for multi-language text recognition.

### Search Algorithm
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Converts text metadata into vectorized form.
- **Cosine Similarity:** Measures query relevance and ranks image search results.

## Implementation Details

![](/res/fotofind_page-0005.jpg)

### Development Environment and Project info
- **Backend:** Python (Flask, PyTorch, Transformers, Scikit-learn, MySQL Connector)
- **Frontend:** HTML, CSS (Bootstrap)
- **Database:** MySQL

#### Project Structure
```
FotoFind/
├── app.py                # Flask application
├── getfeatures.py        # Feature extraction pipeline
├── templates/
│   ├── gallery.html      # Web UI for browsing images
├── static/
│   ├── css/              # Styling files
├── images/               # Directory for uploaded images
├── requirements.txt      # Dependencies
```

#### Usage
```
Just simply clone the repo and run the 'app.py'
Now you can access the application in your favourite browser. -> localhost:5000

!! Ensure that you have installed all required packages. see: requirements.txt (NA yet) 
```

### Workflow
1. **Upload:** Users upload images via the web interface.
2. **Feature Extraction:** Object detection, captioning, and OCR processing.
3. **Metadata Storage:** Extracted data is stored in MySQL.
4. **Search Query:** User searches are matched with stored metadata.
5. **Image Retrieval:** Results are ranked based on relevance and displayed.

## Results
FotoFind successfully retrieves images based on:
- Object detection (e.g., "stop sign" retrieves relevant traffic images).
- Captioning (e.g., "bus to Navy Pier" finds transportation-related images).
- OCR (e.g., "train to Nagoya" matches embedded text in images).

## Future Work
- **Advanced Indexing:** Integrating FAISS/Elasticsearch for scalable search.
- **Domain Adaptation:** Fine-tuning models for specific industries (e.g., healthcare, retail).
- **Cloud Deployment:** Using Docker/Kubernetes for cloud scalability.
- **User Feedback Mechanism:** Allowing manual corrections to improve model performance.

## Usage References
- **Object Detection:** Faster R-CNN (Torchvision), YOLO (Ultralytics)
- **Image Captioning:** ViT-GPT2, BLIP (Hugging Face Transformers)
- **OCR:** EasyOCR (Deep Learning OCR framework)
- **Search Algorithm:** Scikit-learn (TF-IDF, Cosine Similarity)

---
FotoFind provides a powerful and scalable solution for intelligent image retrieval, leveraging cutting-edge computer vision techniques to enhance searchability and metadata enrichment.

