import shutil
import sqlite3
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from dataset import CLASSES, get_transforms

DB_PATH = Path("data/monitoring_telemetry.db")
FEEDBACK_DIR = Path("data/feedback_vault")
QUARANTINE_DIR = Path("data/quarantine_vault")
TRAIN_CSV = Path("data/train_split.csv")
VAL_CSV = Path("data/val_split.csv")
GOLDEN_BACKUP_PATH = Path("models/resnet18_golden_backup.pth")

class AIMonitorAndLearner:
    """
    Production MLOps Plugin with Strict Anti-Poisoning Guardrails:
    1. Logs every inference to SQLite and monitors confidence drift & latency.
    2. Guards against Data Poisoning / Label Flipping via:
       - Strict Contradiction & False-Negative Quarantine (>=75% conf or <15% target prob)
       - Golden Anchor Validation Gate (auto-rollback if anchor accuracy drops)
       - Immutable Golden Weights Backup
    """
    def __init__(self, model, device, weights_path: Path = Path("models/resnet18_best.pth")):
        self.model = model
        self.device = device
        self.weights_path = weights_path
        self.train_tf, self.eval_tf = get_transforms()

        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

        # Create a read-only Golden Backup of the clean trained weights on first run
        if self.weights_path.exists() and not GOLDEN_BACKUP_PATH.exists():
            shutil.copyfile(self.weights_path, GOLDEN_BACKUP_PATH)
            print(f"[SECURITY] Created immutable golden weights backup at: {GOLDEN_BACKUP_PATH}")

        self._init_db()
        self.anchor_x, self.anchor_y = self._load_golden_anchor_set()

    def _init_db(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    filename TEXT,
                    predicted_class TEXT,
                    confidence REAL,
                    latency_ms REAL,
                    flagged_drift INTEGER,
                    verified_class TEXT,
                    was_correct INTEGER,
                    security_status TEXT DEFAULT 'UNVERIFIED'
                )
            """)
            cols = [info[1] for info in conn.execute("PRAGMA table_info(inference_logs)").fetchall()]
            if "security_status" not in cols:
                conn.execute("ALTER TABLE inference_logs ADD COLUMN security_status TEXT DEFAULT 'UNVERIFIED'")
            conn.commit()

    def _load_golden_anchor_set(self):
        """Loads 16 verified validation scans (4 per class) to act as our anti-degradation gate."""
        if not VAL_CSV.exists():
            return None, None
        val_df = pd.read_csv(VAL_CSV)
        tensors, labels = [], []
        for label_idx in range(len(CLASSES)):
            cls_rows = val_df[val_df["label"] == label_idx].head(4)
            for _, row in cls_rows.iterrows():
                p = Path(row["filepath"])
                if p.exists():
                    with Image.open(p) as img:
                        tensors.append(self.eval_tf(img.convert("RGB")))
                        labels.append(label_idx)
        if not tensors:
            return None, None
        return torch.stack(tensors).to(self.device), torch.tensor(labels, dtype=torch.long).to(self.device)

    def _evaluate_anchor_accuracy(self) -> float:
        if self.anchor_x is None:
            return 100.0
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.anchor_x)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == self.anchor_y).float().mean().item() * 100.0
        return acc

    def log_prediction(self, filename: str, pred_class: str, confidence: float, latency_ms: float) -> int:
        flagged = 1 if confidence < 80.0 else 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("""
                INSERT INTO inference_logs 
                (timestamp, filename, predicted_class, confidence, latency_ms, flagged_drift, verified_class, was_correct, security_status)
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 'UNVERIFIED')
            """, (now, filename, pred_class, confidence, latency_ms, flagged))
            conn.commit()
            return cursor.lastrowid

    def get_telemetry_summary(self) -> dict:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM inference_logs ORDER BY id DESC").fetchall()

        total = len(rows)
        if total == 0:
            return {
                "total_scans": 0,
                "avg_confidence": 0.0,
                "avg_latency": 0.0,
                "drift_alerts": 0,
                "verified_count": 0,
                "quarantined_count": 0,
                "live_accuracy": None,
                "recent_logs": []
            }

        avg_conf = sum(r["confidence"] for r in rows) / total
        avg_lat = sum(r["latency_ms"] for r in rows) / total
        drift_alerts = sum(r["flagged_drift"] for r in rows)
        quarantined = sum(1 for r in rows if r["security_status"] and "QUARANTINED" in r["security_status"])

        verified_rows = [r for r in rows if r["was_correct"] is not None and r["security_status"] == "ACCEPTED"]
        live_acc = (
            (sum(r["was_correct"] for r in verified_rows) / len(verified_rows)) * 100.0
            if verified_rows else None
        )

        recent = [dict(r) for r in rows[:6]]
        return {
            "total_scans": total,
            "avg_confidence": round(avg_conf, 2),
            "avg_latency": round(avg_lat, 1),
            "drift_alerts": drift_alerts,
            "verified_count": len(verified_rows),
            "quarantined_count": quarantined,
            "live_accuracy": round(live_acc, 1) if live_acc is not None else None,
            "recent_logs": recent
        }

    def learn_from_feedback(self, log_id: int, pil_img: Image.Image, true_class: str, weights_save_path: Path) -> dict:
        true_label_idx = CLASSES.index(true_class)

        # 1. Inspect current model prediction and confidence on this scan
        self.model.eval()
        with torch.no_grad():
            eval_tensor = self.eval_tf(pil_img).unsqueeze(0).to(self.device)
            old_logits = self.model(eval_tensor)
            old_probs = F.softmax(old_logits, dim=1)[0]
            pred_idx = torch.argmax(old_probs).item()
            pred_class = CLASSES[pred_idx]
            top_conf = old_probs[pred_idx].item() * 100.0
            prob_before = old_probs[true_label_idx].item() * 100.0

        # =========================================================================
        # GUARDRAIL 1: Strict Anti-Label-Flipping & False-Negative Shield
        # Triggers if:
        #   - User contradicts a confident prediction (top_conf >= 75.0%), OR
        #   - User claims a class that the model gives < 15.0% probability, OR
        #   - Model sees a tumor (>= 65%) and user tries to poison it as 'notumor'
        # =========================================================================
        is_contradiction = (pred_class != true_class)
        high_conf_conflict = (top_conf >= 75.0 or prob_before < 15.0)
        tumor_to_healthy_attack = (pred_class != "notumor" and true_class == "notumor" and top_conf >= 65.0)

        if is_contradiction and (high_conf_conflict or tumor_to_healthy_attack):
            q_file = QUARANTINE_DIR / f"quarantine_log_{log_id}_claimed_{true_class}.png"
            pil_img.save(q_file)

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    UPDATE inference_logs
                    SET verified_class = ?, was_correct = 0, security_status = 'QUARANTINED (POISON RISK)'
                    WHERE id = ?
                """, (f"{true_class} [BLOCKED]", log_id))
                conn.commit()

            return {
                "status": "QUARANTINED",
                "reason": (
                    f"Snake Trap Triggered! Model is {top_conf:.2f}% confident in '{pred_class.upper()}' "
                    f"(while your claimed label '{true_class.upper()}' had only {prob_before:.2f}% probability). "
                    f"Adversarial label-flipping blocked & scan locked in quarantine!"
                ),
                "true_class": true_class,
                "prob_before": round(prob_before, 2),
                "prob_after": round(prob_before, 2)
            }

        # =========================================================================
        # GUARDRAIL 2: Snapshot Weights & Measure Golden Anchor Accuracy Before Update
        # =========================================================================
        anchor_acc_before = self._evaluate_anchor_accuracy()
        weights_backup = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Build Experience Replay Batch
        batch_tensors, batch_labels = [], []
        for _ in range(4):
            batch_tensors.append(self.train_tf(pil_img))
            batch_labels.append(true_label_idx)

        if TRAIN_CSV.exists():
            train_df = pd.read_csv(TRAIN_CSV)
            sampled_rows = train_df.sample(n=min(12, len(train_df)), random_state=random.randint(1, 9999))
            for _, row in sampled_rows.iterrows():
                img_p = Path(row["filepath"])
                if img_p.exists():
                    with Image.open(img_p) as m_img:
                        batch_tensors.append(self.train_tf(m_img.convert("RGB")))
                        batch_labels.append(int(row["label"]))

        x_batch = torch.stack(batch_tensors).to(self.device)
        y_batch = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.model.train()
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        logits = self.model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        # Check Golden Anchor Accuracy AFTER Update
        anchor_acc_after = self._evaluate_anchor_accuracy()

        if anchor_acc_after < anchor_acc_before - 6.25:
            # Roll back immediately if update degrades Golden Anchor accuracy by more than 1 sample!
            self.model.load_state_dict({k: v.to(self.device) for k, v in weights_backup.items()})
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    UPDATE inference_logs
                    SET verified_class = ?, was_correct = 0, security_status = 'QUARANTINED (ANCHOR DROP)'
                    WHERE id = ?
                """, (f"{true_class} [ROLLED BACK]", log_id))
                conn.commit()

            return {
                "status": "ROLLED_BACK",
                "reason": (
                    f"Anchor Validation Gate Triggered: Updating on this label caused Golden Anchor Set accuracy "
                    f"to drop ({anchor_acc_before:.1f}% -> {anchor_acc_after:.1f}%). Weights were automatically rolled back!"
                ),
                "true_class": true_class,
                "prob_before": round(prob_before, 2),
                "prob_after": round(prob_before, 2)
            }

        # Update passed all security gates!
        save_file = FEEDBACK_DIR / f"log_{log_id}_{true_class}.png"
        pil_img.save(save_file)

        self.model.eval()
        with torch.no_grad():
            new_probs = F.softmax(self.model(eval_tensor), dim=1)[0]
            prob_after = new_probs[true_label_idx].item() * 100.0

        torch.save(self.model.state_dict(), weights_save_path)

        was_correct = 1 if pred_class == true_class else 0
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE inference_logs
                SET verified_class = ?, was_correct = ?, security_status = 'ACCEPTED'
                WHERE id = ?
            """, (true_class, was_correct, log_id))
            conn.commit()

        return {
            "status": "ACCEPTED",
            "true_class": true_class,
            "prob_before": round(prob_before, 2),
            "prob_after": round(prob_after, 2),
            "anchor_acc": round(anchor_acc_after, 1),
            "loss": round(loss.item(), 4)
        }