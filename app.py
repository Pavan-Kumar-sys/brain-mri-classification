import io
import os
import gc
import base64
import time
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

# Limit PyTorch CPU threads so RAM stays under ~230MB on Render Free Tier
torch.set_num_threads(1)

import sys
sys.path.append(str(Path(__file__).parent / "src"))
from dataset import get_transforms, CLASSES, IMG_SIZE
from models import build_transfer_model, count_parameters
from explain import GradCAM
from monitor_plugin import AIMonitorAndLearner

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = Path("models/resnet18_best.pth")
_, EVAL_TRANSFORM = get_transforms()

print(f"Loading ResNet-18 weights from {WEIGHTS_PATH} on {DEVICE}...")
MODEL = build_transfer_model(num_classes=4, freeze_early_layers=False)
MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
MODEL.to(DEVICE)
MODEL.eval()

for param in MODEL.layer4.parameters():
    param.requires_grad = True

GRAD_CAM = GradCAM(MODEL, target_layer=MODEL.layer4[-1])
_, TOTAL_PARAMS = count_parameters(MODEL)

# Initialize our AI Monitoring & Active Learning Plugin
MONITOR = AIMonitorAndLearner(MODEL, DEVICE, WEIGHTS_PATH)

# Store latest uploaded PIL image in memory for instant Active Learning feedback
LAST_UPLOAD_CACHE = {"log_id": None, "image": None}

MODEL_BENCHMARKS = {
    "model_name": "ResNet-18 (Transfer Learning)",
    "total_params": f"{TOTAL_PARAMS:,}",
    "test_samples": 1468,
    "test_accuracy": 93.12,
    "macro_precision": 93.28,
    "macro_recall": 93.10,
    "macro_f1": 92.78,
    "macro_roc_auc": 98.58,
    "baseline_accuracy": 81.81,
    "baseline_f1": 80.09,
    "per_class": [
        {"name": "glioma", "precision": 99.00, "recall": 78.36, "f1": 87.48, "support": 379},
        {"name": "meningioma", "precision": 87.65, "recall": 95.30, "f1": 91.32, "support": 298},
        {"name": "notumor", "precision": 89.89, "recall": 100.00, "f1": 94.67, "support": 391},
        {"name": "pituitary", "precision": 96.58, "recall": 98.75, "f1": 97.65, "support": 400},
    ]
}

def pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def make_gradcam_overlay(orig_pil: Image.Image, cam_2d: np.ndarray, alpha: float = 0.45) -> Image.Image:
    orig_resized = orig_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    orig_np = np.array(orig_resized) / 255.0

    cam_pil = Image.fromarray(np.uint8(255 * cam_2d)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    cam_resized = np.array(cam_pil) / 255.0

    heatmap_rgb = cm.jet(cam_resized)[:, :, :3]
    blended = np.clip((1 - alpha) * orig_np + alpha * heatmap_rgb, 0, 1)
    return Image.fromarray(np.uint8(255 * blended))

@app.route("/", methods=["GET"])
def index():
    telemetry = MONITOR.get_telemetry_summary()
    return render_template("index.html", benchmarks=MODEL_BENCHMARKS, telemetry=telemetry)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename selected."}), 400

    try:
        t0 = time.perf_counter()
        raw_img = Image.open(file.stream).convert("RGB")
        input_tensor = EVAL_TRANSFORM(raw_img).unsqueeze(0).to(DEVICE)

        MODEL.eval()
        for param in MODEL.layer4.parameters():
            param.requires_grad = True

        MODEL.zero_grad()
        logits = MODEL(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

        cam_2d, pred_idx, top_conf = GRAD_CAM.generate_heatmap(input_tensor)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        pred_class = CLASSES[pred_idx]
        log_id = MONITOR.log_prediction(file.filename, pred_class, round(top_conf, 2), round(inference_ms, 1))

        # Cache for Active Learning
        LAST_UPLOAD_CACHE["log_id"] = log_id
        LAST_UPLOAD_CACHE["image"] = raw_img

        orig_display = raw_img.resize((IMG_SIZE, IMG_SIZE))
        gradcam_display = make_gradcam_overlay(raw_img, cam_2d)

        prob_dict = {
            CLASSES[i]: round(probs[i].item() * 100.0, 2)
            for i in range(len(CLASSES))
        }

        cls_metrics = next(c for c in MODEL_BENCHMARKS["per_class"] if c["name"] == pred_class)

        response_data = {
            "log_id": log_id,
            "predicted_class": pred_class,
            "confidence": round(top_conf, 2),
            "low_confidence_warning": top_conf < 80.0,
            "inference_ms": round(inference_ms, 1),
            "probabilities": prob_dict,
            "class_test_metrics": cls_metrics,
            "original_image": pil_to_base64(orig_display),
            "gradcam_image": pil_to_base64(gradcam_display),
            "telemetry": MONITOR.get_telemetry_summary()
        }
        gc.collect()
        return jsonify(response_data)

    except Exception as e:
        gc.collect()
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    true_class = data.get("true_class")
    log_id = LAST_UPLOAD_CACHE.get("log_id")
    pil_img = LAST_UPLOAD_CACHE.get("image")

    if not true_class or true_class not in CLASSES:
        return jsonify({"error": "Invalid class selected."}), 400
    if log_id is None or pil_img is None:
        return jsonify({"error": "No recent scan in memory to train on."}), 400

    try:
        update_stats = MONITOR.learn_from_feedback(log_id, pil_img, true_class, WEIGHTS_PATH)
        gc.collect()
        return jsonify({
            "update_stats": update_stats,
            "telemetry": MONITOR.get_telemetry_summary()
        })
    except Exception as e:
        gc.collect()
        return jsonify({"error": f"Active learning update failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)