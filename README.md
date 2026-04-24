# carbon-estimation-system
This repository contains the data and machine learning pipelines for an Automated Vegetation Loss Estimator (AVLE). It leverages Google Earth Engine, Sentinel-2 imagery, and a custom U-Net architecture to detect deforestation, calculate localized carbon release, and utilize LLMs to recommend mitigation actions.

# 🌍 AI-Powered Land-Use Change & Carbon Estimation Pipeline

## 📖 Overview
This repository contains the Data Engineering and Machine Learning pipelines for the **Automated Vegetation Loss Estimator (AVLE)**. 

The system automates the extraction of multi-spectral satellite imagery, trains a deep learning segmentation model to detect deforestation, and utilizes a custom mathematical engine to quantify localized carbon emissions resulting from land-use change. This pipeline serves as the core intelligence layer before data is sent to the backend API and verified on the blockchain.

## 🧠 Model Architecture & Tech Stack
* **Data Sourcing:** Google Earth Engine (GEE) Python API
* **Optical Imagery:** Sentinel-2 Surface Reflectance (10m resolution, Cloud-masked)
* **Ground-Truth Labels:** Google Dynamic World (Tree probability masks)
* **Deep Learning Framework:** PyTorch & Segmentation Models PyTorch (SMP)
* **Architecture:** U-Net with a `resnet34` backbone (pre-trained on ImageNet)
* **Geospatial Processing:** `rasterio`, `geemap`, `numpy`

## ⚙️ Pipeline Workflow

Our Colab-based pipeline executes the end-to-end process in four distinct phases:

### 1. Data Acquisition & Cloud Masking
We query Sentinel-2 harmonized datasets over a defined Region of Interest (ROI) for a specific timeframe. A custom bitmasking function is applied to the `QA60` band to remove opaque and cirrus clouds. We calculate the median pixel values across the temporal stack to generate a single, cloud-free composite.

### 2. Ground-Truth Label Generation
We fetch the Dynamic World dataset for the exact same ROI and timeframe. We extract the `trees` classification band and apply a strict `> 0.5` probability threshold to generate binary vegetation masks (1 = Vegetation, 0 = Non-Vegetation). Both the optical and label arrays are exported as aligned GeoTIFFs to Google Drive.

### 3. Pre-processing & U-Net Training
Large GeoTIFFs are sliced into `256x256` pixel patches using `rasterio` to fit into GPU memory. Empty boundary patches are discarded. The data is converted into PyTorch Tensors and fed into a U-Net model using Binary Cross Entropy (BCE) Loss and the Adam optimizer. The final model weights are saved as a `.pth` file.

### 4. AVLE & Carbon Math Engine
The pipeline runs a simulated inference to test the core logic:
* **Change Detection:** Isolates pixels where vegetation existed at Time 1 but not Time 2.
* **AVLE Score:** Incorporates the U-Net's confidence probability and localized region weights.
* **Carbon Estimation:** Converts the 10m pixel resolution into hectares ($1 \text{ pixel} = 0.01 \text{ hectares}$) and applies a localized Above-Ground Biomass (AGB) emission factor to calculate total Tons of CO2 released.

## 🚀 How to Run the Pipeline

**Prerequisites:**
You must have a Google Cloud Project with the **Earth Engine API** enabled and registered for non-commercial/academic use.

1. Open `Carbon_Estimation_Pipeline.ipynb` in Google Colab.
2. Run **Cell 1** to install dependencies (`earthengine-api`, `rasterio`, `segmentation-models-pytorch`).
3. Run **Cell 2** to authenticate your Google Cloud Project. **Replace `your-gcp-project-id` with your actual project ID.**
4. Wait 5-10 minutes for Google's servers to export the `.tif` files to your Google Drive.
5. Run **Cell 3** to slice the images and train the U-Net.
6. Run **Cell 4** to execute the AVLE carbon math logic.

## ⚠️ Important Note on Large Files
To maintain repository health, the following files are strictly ignored via `.gitignore` and **are not hosted on GitHub**:
* Raw satellite imagery (`.tif` files)
* Trained model weights (`unet_vegetation_weights.pth`)

If you are a developer cloning this repo for the backend API, you must request the `.pth` file from the ML team and place it manually in the `backend/` directory.
