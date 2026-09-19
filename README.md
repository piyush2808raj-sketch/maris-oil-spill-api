# MARIS — AI-Based Oil Spill Detection API

MARIS (Maritime AI for Remote-sensing Intelligence System) is an AI-powered API for detecting and segmenting oil spills in satellite imagery using deep learning.

The system uses Sentinel-1 SAR imagery and a two-stage AI pipeline:

**Classification → Segmentation → Spill Analysis**

## 🚀 Features

- 🛰️ Oil spill detection from Sentinel-1 SAR imagery
- 🤖 Oil / No-Oil image classification
- 🎯 Pixel-level oil spill segmentation
- 📊 Spill area and area percentage estimation
- 📍 Spill centroid extraction
- 📦 Bounding box extraction
- 📐 Spill perimeter calculation
- 🖼️ Segmentation mask and overlay generation
- 🔌 REST API using FastAPI
- 📚 Interactive Swagger API documentation
- ☁️ Cloud-deployed API

## 🧠 How It Works

MARIS uses a two-stage deep learning pipeline to detect and analyze possible oil spills.


Satellite SAR Image
        ↓
Image Preprocessing
        ↓
Oil Spill Classification
        ↓
   Oil Detected?
      /      \
    No        Yes
    ↓          ↓
No Spill   Segmentation
              ↓
        Spill Mask
              ↓
       Spill Analysis
              ↓
        JSON Response

## 🛰️ Satellite Data

MARIS uses **Synthetic Aperture Radar (SAR)** satellite imagery for oil spill detection and segmentation.

The project primarily uses **Sentinel-1 SAR imagery**. The satellite images contain **grayscale radar backscatter information**, which helps identify surface features on the ocean that may correspond to oil spills.

SAR is particularly useful for maritime monitoring because it can capture images **during both day and night** and is not dependent on sunlight or clear-sky conditions.

> **Note:** The current MARIS demo does not use a separately provided VV/VH polarization dataset.

## 🤖 Deep Learning Models

MARIS uses two separate **TensorFlow/Keras deep learning models** for oil spill analysis:

### 🔹 **1. Oil Spill Classification Model**

The classification model determines whether the input SAR image contains an **oil spill or no oil spill**.

- **Architecture:** MobileNetV2 with fine-tuning
- **Input:** 224 × 224 grayscale SAR image
- **Output:** Oil Spill / No Oil Spill
- **Purpose:** Initial screening of the satellite image
- **Saved Model:** `oil_spill_model.keras`

### 🔹 **2. Oil Spill Segmentation Model**

The segmentation model identifies the **exact region of the detected oil spill** at the pixel level.

- **Architecture:** U-Net
- **Input:** SAR image
- **Output:** Pixel-level oil spill mask
- **Purpose:** Locate and delineate the detected oil spill
- **Saved Model:** `best_unet_oilspill.keras`

### 🔄 **Model Pipeline**

The two models work sequentially:

**SAR Image → Classification → Oil Spill Detected → Segmentation → Spill Mask → Spill Measurements**

If the classification model predicts **No Oil Spill**, the segmentation stage is skipped.

## 📊 Datasets & Attribution

MARIS uses publicly available **SAR oil spill datasets** for training and evaluation.

### 🔹 **1. Sentinel-1 SAR Oil Spill Detection Dataset**

Used primarily for **oil spill classification** using Sentinel-1 SAR imagery.

- **Source:** Kaggle
- **Dataset:** Sentinel-1 SAR Oil Spill Detection Dataset
- **Image Type:** Grayscale SAR images
- **Classes:** Oil Spill / No Oil Spill

🔗 **Dataset:** https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset

### 🔹 **2. Deep-SAR SOS Oil Spill Detection Dataset**

Used for **oil spill segmentation** with pixel-level masks.

- **Source:** Kaggle
- **Dataset:** Deep-SAR SOS Oil Spill Detection Dataset
- **Satellite Data:** Sentinel-1 and PALSAR SAR imagery
- **Labels:** Pixel-level segmentation masks

🔗 **Dataset:** https://www.kaggle.com/datasets/bitsandlayers/sar-oil-spill-segmentation-dataset-sos

### 📌 **Attribution**

The datasets are publicly available and are used for **research and educational purposes**. Dataset ownership and licensing remain with the respective original creators and sources.

## 📁 Project Structure

MARIS/
│
├── ai_models/
│   ├── classification_model/
│   │   └── oil_spill_model.keras
│   │
│   └── segmentation_model/
│       └── best_unet_oilspill.keras
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── inference/
│   ├── input_images/
│   ├── output/
│   ├── preprocessing/
│   ├── postprocessing/
│   └── pipeline/
│
├── training/
│   ├── classification_training.ipynb
│   └── segmentation_training.ipynb
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md

## 🛠️ Tech Stack

### 🔹 **Machine Learning & Deep Learning**
- **Python**
- **TensorFlow / Keras**
- **MobileNetV2** — Image classification
- **U-Net** — Oil spill segmentation
- **NumPy** — Numerical processing
- **Pillow (PIL)** — Image loading and processing


### 🔹 **API & Backend**
- **FastAPI** — REST API development
- **Uvicorn** — API server

### 🔹 **Deployment**
- **Docker** — Containerization
- **Render** — Cloud deployment

### 🔹 **Development & Training**
- **Google Colab** — Model training
- **Jupyter Notebook** — Training experiments
- **Git & GitHub** — Version control and project management

## ☁️ Deployment

The MARIS API is containerized using **Docker** and deployed on **Render** as a cloud-based FastAPI service.

### 🔹 **Deployment Platform**
- **Platform:** Render
- **Containerization:** Docker
- **Backend:** FastAPI
- **Server:** Uvicorn

### 🔗 **Live API**

The deployed API provides endpoints for submitting SAR images and receiving oil spill detection and segmentation results.

- **API:** `https://maris-oil-spill-api.onrender.com`
- **Swagger Documentation:** `https://maris-oil-spill-api.onrender.com/docs`
- **Health Check:** `https://maris-oil-spill-api.onrender.com/health`

## 🔌 API Endpoints

The MARIS API provides endpoints for health monitoring, oil spill prediction, and interactive API documentation.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Checks whether the API is running |
| `POST` | `/predict` | Uploads a SAR image and performs oil spill detection and segmentation |
| `GET` | `/docs` | Opens the interactive Swagger API documentation |

### 🔹 **Prediction Flow**

The `/predict` endpoint follows this pipeline:

**Input SAR Image → Preprocessing → Classification → Segmentation → Postprocessing → JSON Response**

If no oil spill is detected during classification, the segmentation stage is skipped.

## 📚 API Documentation

MARIS provides interactive API documentation through **Swagger UI**, allowing users to test the API endpoints directly from their browser.

### 🔹 **Swagger UI**

Open:

`https://maris-oil-spill-api.onrender.com/docs`

From the Swagger interface, users can:

- Upload a SAR image
- Send a prediction request
- View the API response
- Test the `/health` endpoint
- Explore the available API schemas

## 📦 Example API Request

The `/predict` endpoint accepts a SAR satellite image as a file upload.

Users can upload an image through the interactive **Swagger UI**:

`https://maris-oil-spill-api.onrender.com/docs`

The uploaded image is processed by the MARIS inference pipeline, which performs classification and, when an oil spill is detected, segmentation.

## 📦 Example API Response

The `/predict` endpoint returns the oil spill analysis results along with the generated **segmentation mask** and **overlay image**.

### 🔹 **Response**

The API response contains:

- **JSON results** — Oil spill prediction, confidence, and extracted spill measurements.
- **Segmentation Mask** — Pixel-level mask of the detected oil spill.
- **Overlay Image** — Predicted spill region overlaid on the original SAR image.

### 🔹 **Example Response**

{
  "job_id": "703a83c3",
  "result": {
    "oil_probability": 0.6650543808937073,
    "spill_area": {
      "oil_pixels": 9290,
      "area_percentage": 14.1754150390625
    },
    "centroid": {
      "x": 81.19913885898816,
      "y": 168.0832077502691
    },
    "bounding_box": {
      "x_min": 0,
      "y_min": 38,
      "x_max": 168,
      "y_max": 255
    },
    "perimeter": 890,
    "job_id": "703a83c3",
    "result": "OIL_DETECTED"
  },
  "files": {
    "mask": "https://maris-oil-spill-api.onrender.com/result/703a83c3/predicted_mask.png",
    "overlay": "https://maris-oil-spill-api.onrender.com/result/703a83c3/overlay.png",
    "json": "https://maris-oil-spill-api.onrender.com/result/703a83c3/result.json"
  }
}

## 🖼️ Outputs

MARIS produces both **visual and numerical outputs** from the uploaded SAR image.

### 🔹 **Visual Outputs**

- **Original SAR Image** — The input satellite image.
- **Segmentation Mask** — Pixel-level mask showing the detected oil spill region.
- **Overlay Image** — The predicted spill region overlaid on the original SAR image.

### 🔹 **Numerical Outputs**

- **Oil Spill Prediction** — Oil Spill / No Oil Spill
- **Confidence Score** — Model confidence for the prediction
- **Spill Area** — Estimated area of the detected spill region
- **Area Percentage** — Percentage of the image covered by the spill
- **Centroid** — Center location of the detected spill region
- **Bounding Box** — Coordinates surrounding the detected spill
- **Perimeter** — Perimeter of the detected spill region

## 🎯 Project Goal

The goal of MARIS is to provide an AI-based system for **automated oil spill detection and segmentation from SAR satellite imagery**.

The API is designed to transform a satellite image into actionable information by:

- Detecting whether an oil spill is present.
- Identifying the affected region at the pixel level.
- Generating segmentation masks and visual overlays.
- Extracting quantitative spill measurements.
- Providing the results through an accessible REST API.

This AI detection module forms the **DETECT** stage of the broader MARIS concept for maritime oil spill monitoring.

## 🔮 Future Improvements

- Improve model performance with larger and more diverse SAR datasets.
- Support additional SAR satellite sources and polarization information.
- Improve segmentation accuracy for complex ocean conditions and look-alikes.
- Add more advanced spill shape and boundary analysis.
- Integrate real-time satellite data for automated monitoring.
- Connect the detection system with **oil spill drift/hindcasting models**.
- Integrate **AIS vessel data** for potential source-vessel attribution.
- Add uncertainty estimation to the detection and segmentation results.
- Develop a more interactive visualization interface for detected spills.

## 📄 License

This project is intended for **research and educational purposes**.

The datasets used by MARIS are subject to their respective licenses and attribution requirements. Please refer to the original dataset sources for their specific licensing terms.
