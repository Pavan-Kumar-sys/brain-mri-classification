# 🧠 Brain MRI Tumor Classifier & Guarded MLOps Dashboard

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://docker.com)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7.svg)](https://render.com)

A production-ready Deep Learning and MLOps pipeline for Brain MRI Tumor Classification (Glioma, Meningioma, Pituitary, No Tumor). This project demonstrates how to transition a model from a local Jupyter Notebook into a secure, cloud-hosted production environment with real-time explainability and active learning security.

🔗 **Live Application:** [https://brain-mri-classification-2pqz.onrender.com/]  
📄 **Full Technical Report:** [Read the `brain_mri_mlops_report.pdf`](brain_mri_mlops_report.pdf) for mathematical formulas, architecture details, and full model benchmarks.  
🧪 **Test It Yourself:** Don't have an MRI scan handy? Use the images provided in the [`sample_scans/`](sample_scans/) directory to test the live app!

---

## 🚀 Key Engineering Features

* **Deep Learning Engine:** Fine-tuned PyTorch ResNet-18 (Transfer Learning) achieving **93.12% held-out test accuracy** and **98.58% ROC-AUC** across 7,000+ deduplicated scans.
* **Explainable AI (XAI):** Real-time **Grad-CAM heatmaps** extracted dynamically from `layer4` to visually validate model attention before trusting predictions.
* **Guarded Active Learning ("Snake Trap" 🐍):** A Human-in-the-Loop experience-replay engine wrapped in a 3-layer security gate. It automatically blocks adversarial label-flipping (e.g., trying to poison the model by labeling a clear tumor as "No Tumor"), quarantines attacks, and rolls back weights if the Golden Anchor validation set degrades.
* **Cloud-Optimized MLOps:** A full Flask + SQLite telemetry dashboard tracking inference latency and confidence drift. Containerized via Docker with extreme memory optimizations (head-only micro-batch tuning, thread capping, and explicit garbage collection) to run flawlessly inside a strict **512MB RAM cloud constraint**.

---

## 📊 Model Benchmarks

| Metric | Score (Test Set) |
| :--- | :--- |
| **Accuracy** | 93.12% (+11.31% over baseline CNN) |
| **Macro Precision** | 93.28% |
| **Macro Recall** | 93.10% |
| **Macro F1-Score** | 92.78% |
| **Macro ROC-AUC** | 98.58% |

---

## 🛠️ Tech Stack

* **Machine Learning:** PyTorch, Torchvision, Scikit-Learn, Pillow, NumPy, Pandas
* **Web / MLOps Dashboard:** Flask, SQLite, Tailwind CSS, Gunicorn
* **Deployment:** Docker, Render (Free Tier Optimized)

---

## 💻 Running the Project Locally

### 1. Clone and Install
```bash
git clone [https://github.com/Pavan-Kumar-sys/brain-mri-classification.git](https://github.com/Pavan-Kumar-sys/brain-mri-classification.git)
cd brain-mri-classification
python -m venv venv
source venv/Scripts/activate  # On Windows
pip install -r requirements.txt
