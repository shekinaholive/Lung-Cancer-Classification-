# Lung Cancer Subtype Classification
### MobileNetV2 feature extraction + XGBoost classification + explainability

An academic computer-vision prototype for classifying histopathology images into three categories from the **LC25000** dataset. The project combines a pretrained convolutional neural network (CNN) feature extractor with a gradient-boosted classifier and explores model explanations using Grad-CAM and SHAP.

**Focus:** Hybrid machine learning · Histopathology · Explainable AI · Gradio

## What this project does

- Extracts image features using **ImageNet-pretrained MobileNetV2** (without its classification head; global average pooling).
- Trains an **XGBoost** classifier on the extracted features.
- Evaluates three classes: **lung_aca** (lung adenocarcinoma), **lung_n** (non-cancerous lung tissue) and **lung_scc** (lung squamous cell carcinoma).
- Explores prediction explanations in a separate **Grad-CAM + SHAP** notebook.
- Includes a **Gradio** notebook for image upload and interactive predictions.

## Architecture

~~~mermaid
flowchart LR
    A["LC25000 image"] --> B["Resize to 224 × 224"]
    B --> C["MobileNetV2 feature extractor"]
    C --> D["Feature vector"]
    D --> E["XGBoost classifier"]
    E --> F["Three-class prediction"]
    C -.-> G["Explainability analysis"]
    E -.-> G
~~~

The MobileNetV2 feature extractor is frozen in the hybrid training notebook. XGBoost learns the classification step from the extracted image features; it is **not** trained end-to-end jointly with MobileNetV2.

## Dataset and evaluation

The notebooks use separate **train**, **val** and **test** directories of histopathology images. The recorded run reports:

| Partition | Images |
| --- | ---: |
| Training | 12,008 |
| Validation | 1,501 |
| Test | 1,501 |

**Results recorded in Hybrid_Model.ipynb**

| Metric | Recorded result |
| --- | --- |
| Test accuracy | **0.98** (rounded in the classification report) |
| Validation accuracy | **0.98** (rounded in the classification report) |
| Test macro F1-score | **0.98** |
| Test evaluation samples | **1,501** |

The test confusion matrix printed in the notebook is:

~~~text
[[489,   1,  11],
 [  0, 500,   0],
 [ 14,   0, 486]]
~~~

A separate baseline CNN comparison is described in the project resume, but its reproducible evaluation is not consolidated in this repository. The table above reports **only metrics visible in the hybrid notebook**.

> These are dataset-specific experimental results, not evidence of clinical diagnostic performance. The repository does not document patient- or slide-level external validation.

## Repository guide

| File | Purpose |
| --- | --- |
| [Hybrid_Model.ipynb](Hybrid_Model.ipynb) | Load image splits, extract MobileNetV2 features, train XGBoost, and evaluate on test/validation sets |
| [Single_model.ipynb](Single_model.ipynb) | Standalone model experiments / comparison |
| [XAI_GradCam_SHAP.ipynb](XAI_GradCam_SHAP.ipynb) | Grad-CAM and SHAP interpretability experiments |
| [Gradio_Web_deployment.ipynb](Gradio_Web_deployment.ipynb) | Interactive image-classification demo notebook |

## Reproducing the hybrid notebook

**Environment:** The saved notebooks are written for **Google Colab** with Google Drive paths. Python packages used include TensorFlow/Keras, XGBoost, scikit-learn, NumPy, Matplotlib, Seaborn and (for the demo) Gradio.

1. Obtain and prepare the LC25000 image dataset under these three folders:
   - **Lung_Cancer_Split/train/** — one subfolder per class
   - **Lung_Cancer_Split/val/** — one subfolder per class
   - **Lung_Cancer_Split/test/** — one subfolder per class
2. Upload **Lung_Cancer_Split/** to Google Drive, or edit **drive_data_path** in **Hybrid_Model.ipynb** to match your environment.
3. Run the hybrid notebook from top to bottom. It saves **mobilenet_features.npy**, **mobilenet_labels.npy**, and **xgboost_hybrid_model.json** to the configured Drive locations.
4. Open **XAI_GradCam_SHAP.ipynb** for explanation experiments, or **Gradio_Web_deployment.ipynb** for the interactive interface. Update model paths before running.

The dataset, trained model weights and Google Drive artifacts are **not included** in this repository. The Gradio notebook loads the saved XGBoost model and constructs a pretrained MobileNetV2 feature extractor.

## Known limitations and next engineering steps

- **Preprocessing consistency:** The hybrid training notebook uses **ImageDataGenerator(rescale=1./255)**, whereas the Gradio demo calls MobileNetV2's **preprocess_input**. These are different transformations. Align them and reevaluate before interpreting demo predictions as equivalent to the reported test results.
- **Validation scope:** No independently held-out clinical cohort or patient-level split audit is included.
- **Reproducibility:** A version-pinned environment, source dataset manifest and script-based inference pipeline would make the work easier to reproduce.
- **Model confidence:** Predicted class probabilities should not be interpreted as calibrated medical confidence.

## Responsible use

**For research and learning only.** This project is not a medical device, is not clinically validated, and must not be used to diagnose or guide treatment.

---
**Author:** [Shekinah Olive](https://github.com/shekinaholive) · [More projects](https://github.com/shekinaholive)
