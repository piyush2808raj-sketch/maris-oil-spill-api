---
title: MARIS Oil Spill Detection API
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# MARIS — Maritime Oil Spill Detection API

AI-powered API for detecting and segmenting oil spills in Sentinel-1 SAR imagery.

## Pipeline

1. Oil Spill Classification
2. Oil Spill Segmentation
3. Spill Area Calculation
4. Centroid Detection
5. Bounding Box Detection
6. Perimeter Calculation

## API

### Health Check

`GET /health`

### Prediction

`POST /predict`

Upload a JPG, JPEG, or PNG SAR image.

### API Documentation

`/docs`