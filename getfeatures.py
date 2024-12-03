# import torch
# import torchvision
# from torchvision import transforms
# from transformers import (
#     VisionEncoderDecoderModel,
#     ViTImageProcessor,
#     AutoTokenizer,
#     BlipProcessor,
#     BlipForConditionalGeneration
# )
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # **Load Models Globally for Efficiency**

# # Load RCNN Model
# rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# rcnn_model.eval()
# transform_rcnn = transforms.Compose([transforms.ToTensor()])

# # Image Captioning (RCNN) Model - ViT-GPT2
# ic_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# ic_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# ic_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ic_model.to(device)

# # YOLO + BLIP Models
# yolo_model = YOLO('yolov8x.pt')  # Pre-trained YOLOv8
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # COCO labels for RCNN
# COCO_INSTANCE_CATEGORY_NAMES = [
#     '_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
#     'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#     'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
#     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
#     'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#     'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
#     'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#     'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
#     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]

# def rcnn_captioning(image):
#     """Detect objects using RCNN and generate a caption using ViT-GPT2."""
#     image_tensor = transform_rcnn(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = rcnn_model(image_tensor)

#     threshold = 0.8
#     scores = outputs[0]['scores'].cpu().numpy()
#     labels = outputs[0]['labels'].cpu().numpy()
#     high_conf_indices = np.where(scores >= threshold)[0].tolist()

#     detected_labels = [COCO_INSTANCE_CATEGORY_NAMES[labels[idx]] for idx in high_conf_indices]

#     # Generate caption using ViT-GPT2
#     pixel_values = ic_processor(images=image, return_tensors="pt").pixel_values.to(device)
#     with torch.no_grad():
#         output_ids = ic_model.generate(pixel_values, max_length=16, num_beams=4)
#     caption = ic_tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     return detected_labels, caption

# def yolo_captioning(image_path, image):
#     """Detect objects using YOLO and generate a caption using BLIP."""
#     # YOLO detection
#     results = yolo_model(image_path)
#     detected_objects = []
#     for box in results[0].boxes:
#         label_index = int(box.cls)  # Label index
#         confidence = float(box.conf)  # Confidence score
#         label_name = results[0].names[label_index]  # Category name
#         detected_objects.append(label_name)

#     # Generate caption using BLIP
#     inputs = blip_processor(images=image, return_tensors="pt").pixel_values.to(device)
#     with torch.no_grad():
#         outputs = blip_model.generate(inputs)
#     blip_caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

#     return detected_objects, blip_caption

# def get_features(image_path):
#     """Process image with RCNN and YOLO, generate labels and captions."""
#     image = Image.open(image_path).convert("RGB")
    
#     # RCNN detection and captioning
#     rcnn_labels, rcnn_caption = rcnn_captioning(image)
    
#     # YOLO detection and BLIP captioning
#     yolo_labels, yolo_caption = yolo_captioning(image_path, image)
    
#     # Combine labels from both models
#     combined_labels = set(rcnn_labels + yolo_labels)
    
#     # Combine captions from both models
#     combined_captions = f"{rcnn_caption}; {yolo_caption}"
    
#     return {
#         'set_labels': list(combined_labels),
#         'captions': combined_captions
#     }

# import torch
# import torchvision
# from torchvision import transforms
# from transformers import (
#     VisionEncoderDecoderModel,
#     ViTImageProcessor,
#     AutoTokenizer,
#     BlipProcessor,
#     BlipForConditionalGeneration
# )
# from ultralytics import YOLO
# from PIL import Image
# import pytesseract
# import numpy as np
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # **Load Models Globally for Efficiency**

# # Load RCNN Model
# rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# rcnn_model.eval()
# transform_rcnn = transforms.Compose([transforms.ToTensor()])

# # Image Captioning (RCNN) Model - ViT-GPT2
# ic_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# ic_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# ic_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ic_model.to(device)

# # YOLO + BLIP Models
# yolo_model = YOLO('yolov8x.pt')  # Pre-trained YOLOv8
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # COCO labels for RCNN
# COCO_INSTANCE_CATEGORY_NAMES = [
#     '_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
#     'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#     'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
#     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
#     'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#     'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
#     'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#     'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
#     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]

# def rcnn_captioning(image):
#     """Detect objects using RCNN and generate a caption using ViT-GPT2."""
#     image_tensor = transform_rcnn(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = rcnn_model(image_tensor)

#     threshold = 0.8
#     scores = outputs[0]['scores'].cpu().numpy()
#     labels = outputs[0]['labels'].cpu().numpy()
#     high_conf_indices = np.where(scores >= threshold)[0].tolist()

#     detected_labels = [COCO_INSTANCE_CATEGORY_NAMES[labels[idx]] for idx in high_conf_indices]

#     # Generate caption using ViT-GPT2
#     pixel_values = ic_processor(images=image, return_tensors="pt").pixel_values.to(device)
#     with torch.no_grad():
#         output_ids = ic_model.generate(pixel_values, max_length=16, num_beams=4)
#     caption = ic_tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     return detected_labels, caption

# def yolo_captioning(image_path, image):
#     """Detect objects using YOLO and generate a caption using BLIP."""
#     # YOLO detection
#     results = yolo_model(image_path)
#     detected_objects = []
#     for box in results[0].boxes:
#         label_index = int(box.cls)  # Label index
#         confidence = float(box.conf)  # Confidence score
#         label_name = results[0].names[label_index]  # Category name
#         detected_objects.append(label_name)

#     # Generate caption using BLIP
#     inputs = blip_processor(images=image, return_tensors="pt").pixel_values.to(device)
#     with torch.no_grad():
#         outputs = blip_model.generate(inputs)
#     blip_caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

#     return detected_objects, blip_caption

# def extract_text_with_ocr(image):
#     """Extract text from an image using Tesseract OCR."""
#     ocr_text = pytesseract.image_to_string(image)
#     return ocr_text.strip()

# def get_features(image_path):
#     """Process image with RCNN, YOLO, and OCR to generate labels and captions."""
#     image = Image.open(image_path).convert("RGB")
    
#     # RCNN detection and captioning
#     rcnn_labels, rcnn_caption = rcnn_captioning(image)
    
#     # YOLO detection and BLIP captioning
#     yolo_labels, yolo_caption = yolo_captioning(image_path, image)
    
#     # OCR text extraction
#     ocr_text = extract_text_with_ocr(image)
#     ocr_part = f"(ocr :> {ocr_text})" if ocr_text else "(ocr :> empty)"
    
#     # Combine labels from both models
#     combined_labels = set(rcnn_labels + yolo_labels)
    
#     # Combine captions from all models
#     combined_captions = f"{rcnn_caption}; {yolo_caption}; {ocr_part}"
    
#     return {
#         'set_labels': list(combined_labels),
#         'captions': combined_captions
#     }


# print(get_features(r"C:\Users\tonyw\Downloads\archivecoco\coco2017\test2017\000000114318.jpg"))


import torch
import torchvision
from torchvision import transforms
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration
)
from ultralytics import YOLO
from PIL import Image
import numpy as np
import warnings
import easyocr

# Suppress warnings
warnings.filterwarnings("ignore")

# **Load Models Globally for Efficiency**

# Load RCNN Model
rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
rcnn_model.eval()
transform_rcnn = transforms.Compose([transforms.ToTensor()])

# Image Captioning (RCNN) Model - ViT-GPT2
ic_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
ic_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
ic_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ic_model.to(device)

# YOLO + BLIP Models
# yolo_model = YOLO('yolov8x.pt')  # Pre-trained YOLOv8
yolo_model = YOLO('yolo11s.pt')  # Pre-trained YOLOv8
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# COCO labels for RCNN
COCO_INSTANCE_CATEGORY_NAMES = [
    '_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def rcnn_captioning(image):
    """Detect objects using RCNN and generate a caption using ViT-GPT2."""
    image_tensor = transform_rcnn(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = rcnn_model(image_tensor)

    threshold = 0.8
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    high_conf_indices = np.where(scores >= threshold)[0].tolist()

    detected_labels = [COCO_INSTANCE_CATEGORY_NAMES[labels[idx]] for idx in high_conf_indices]

    # Generate caption using ViT-GPT2
    pixel_values = ic_processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = ic_model.generate(pixel_values, max_length=16, num_beams=4)
    caption = ic_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return detected_labels, caption

def yolo_captioning(image_path, image):
    """Detect objects using YOLO and generate a caption using BLIP."""
    # YOLO detection
    results = yolo_model(image_path)
    detected_objects = []
    for box in results[0].boxes:
        label_index = int(box.cls)  # Label index
        confidence = float(box.conf)  # Confidence score
        label_name = results[0].names[label_index]  # Category name
        detected_objects.append(label_name)

    # Generate caption using BLIP
    inputs = blip_processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = blip_model.generate(inputs)
    blip_caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

    return detected_objects, blip_caption

def extract_text_with_easyocr(image_path):
    """Extract text from an image using EasyOCR."""
    reader = easyocr.Reader(['en'])  # Specify the language
    results = reader.readtext(image_path)
    extracted_texts = [result[1] for result in results]
    return " ".join(extracted_texts).strip()

def get_features(image_path):
    """Process image with RCNN, YOLO, and OCR to generate labels and captions."""
    image = Image.open(image_path).convert("RGB")
    
    # RCNN detection and captioning
    rcnn_labels, rcnn_caption = rcnn_captioning(image)
    
    # YOLO detection and BLIP captioning
    yolo_labels, yolo_caption = yolo_captioning(image_path, image)
    
    # OCR text extraction using EasyOCR
    ocr_text = extract_text_with_easyocr(image_path)
    ocr_part = f"OCR :> {ocr_text}" if ocr_text else ""
    
    # Combine labels from both models
    combined_labels = set(rcnn_labels + yolo_labels)
    
    # Combine captions from all models
    combined_captions = f"{rcnn_caption}; {yolo_caption}; {ocr_part}"
    
    return {
        'set_labels': list(combined_labels),
        'captions': combined_captions
    }

# Example usage:
print(get_features(r"C:\Users\tonyw\Downloads\archivecoco\coco2017\test2017\000000114318.jpg"))