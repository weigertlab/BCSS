# BCSS Dataset Overview

This repository contains code for downloading the Breast Cancer Semantic Segmentation (BCSS) dataset, which provides annotated histopathology image crops for training machine learning models.

## What the Code Does

The codebase is a Python-based download tool that retrieves:
- **RGB image crops** from whole slide images (WSIs) at specific regions of interest
- **Corresponding segmentation masks** with pixel-level annotations 
- **JSON annotation files** (optional) containing coordinates and metadata

## How Crops Are Downloaded

The crop downloading process works through the following mechanism:

### 1. Data Source and Authentication
- Connects to HistomicsTK server API at `https://demo.kitware.com/histomicstk/api/v1/`
- Uses API key authentication to access the TCGA breast cancer slide collection
- Source folder ID: `5bbdeba3e629140048d017bb`

### 2. Region of Interest (ROI) Selection
- Reads predefined ROI coordinates from `meta/roiBounds.csv`
- Each row contains: slide name, xmin, ymin, xmax, ymax coordinates, and mask download link
- ROIs are rectangular regions within larger whole slide images that contain relevant tissue

### 3. RGB Image Crop Extraction
For each ROI, the system:
- Constructs API requests to the HistomicsTK tile server: `/item/{slide_id}/tiles/region`
- Specifies exact pixel coordinates (left, right, top, bottom) for the crop region
- Downloads crops at configurable resolution:
  - **MPP (Microns Per Pixel)**: Default 0.25 MPP (equivalent to ~40x magnification)
  - **MAG (Magnification)**: Alternative specification method
  - **Base magnification**: If neither MPP nor MAG specified
- Saves RGB crops as PNG files with naming pattern: `{slide_name}_xmin{x}_ymin{y}_{resolution}.png`

### 4. Mask Download and Processing
- Downloads pre-computed segmentation masks from Figshare URLs stored in the ROI bounds CSV
- Each mask is a PNG image where pixel values encode tissue class membership (22 classes total)
- Resizes masks to match the resolution of corresponding RGB crops using nearest-neighbor interpolation
- Ground truth codes are defined in `meta/gtruth_codes.tsv` (tumor=1, stroma=2, etc.)

### 5. File Organization
Downloaded files are organized into directories:
- `images/`: RGB crop images  
- `masks/`: Segmentation masks
- `annotations/`: JSON annotation files (if enabled)
- `logs/`: Download logs and error tracking

### 6. Key Features
- **Batch downloading**: Processes multiple slides automatically
- **Resume capability**: Skips already downloaded slides
- **Flexible resolution**: Configurable MPP/magnification settings
- **Selective downloading**: Can download only specific slides via `SLIDES_TO_KEEP`
- **Pipeline control**: Can enable/disable images, masks, or annotations via `PIPELINE` config

The system essentially extracts standardized tissue crops from large whole slide images, providing both the visual data (RGB) and ground truth labels (masks) needed for supervised learning of histopathology image analysis models.