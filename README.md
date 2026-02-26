# 🏥 SurgAssist — Surgical Instrument Intelligence System

A production-ready computer vision + ML pipeline that detects, tracks, and analyzes surgical instrument usage in real time — built to improve OR efficiency, reduce instrument waste, and predict sterilization bottlenecks.

---

## 🎯 What It Does

| Module | Function |
|---|---|
| **Live Detection** | Detects surgical instruments frame-by-frame using color segmentation + YOLOv8 |
| **SORT Tracker** | Tracks each instrument instance with a unique ID using Kalman filtering |
| **Usage Analytics** | Measures active time, idle time, first appearance, and waste per instrument |
| **Demand Prediction** | Random Forest model predicts which instrument will be needed next |
| **Sterilization Simulator** | Models cleaning queues and flags bottleneck/shortage risk |
| **OR Efficiency Score** | Custom metric (0–100) combining idle ratio and waste penalty |
| **Live Dashboard** | Real-time Streamlit interface with heatmap, stats, and CSV export |

---

## 📊 Dashboard Preview

```
┌─────────────────────────────────────────────────────────┐
│  🏥 SurgAssist — Surgical Instrument Intelligence        │
├──────────────────────────┬──────────────────────────────┤
│  📹 Live Detection Feed  │  📊 Real-Time Stats          │
│  [Video with bboxes]     │  🟢 scalpel   241s  Idle: 0% │
│                          │  🟢 forceps   255s  Idle: 0% │
│                          │  🟢 needle_h  298s  Idle: 0% │
│                          │  🔴 suction   0.0s  Idle:100%│
├──────────────┬───────────┴──────────┬───────────────────┤
│ 🔮 Demand    │ 🧪 Sterilization     │ 📈 OR Efficiency  │
│ Prediction   │ Bottleneck           │ Metrics           │
│ scalpel: 95% │ 🟢 No bottleneck     │ Score: 92.5/100   │
│ needle: 4%   │ Processed: 3         │ Waste:  25.0%     │
│              │ Queue: 0             │ Switches: 18      │
├──────────────┴──────────────────────┴───────────────────┤
│  🗺 Instrument Usage Heatmap                             │
│  [forceps      ████████░░████████████████████████████]  │
│  [needle_holder ████████████░░░████████░░░████████████] │
│  [scalpel      ░░░░░░░░░░░░████████████████████████████]│
└─────────────────────────────────────────────────────────┘
```

---

## 🧱 Tech Stack

- **Python 3.10+**
- **YOLOv8** (Ultralytics) — object detection
- **OpenCV** — computer vision + color segmentation
- **SORT** (Kalman Filter + Hungarian Algorithm) — multi-object tracking
- **PyTorch** — deep learning backbone
- **Scikit-learn** — Random Forest demand predictor
- **SQLite** — lightweight event storage
- **Streamlit** — real-time dashboard
- **Matplotlib** — heatmap visualization
- **Pandas / NumPy** — data processing

---

## 🗂 Project Structure

```
surgassist/
├── config.yaml                  # All system settings
├── main.py                      # Entry point
├── requirements.txt
│
├── data/
│   ├── generate_sample.py       # Synthetic OR video generator
│   └── videos/sample.mp4        # Auto-generated on first run
│
├── detection/
│   └── detector.py              # Color detection + YOLOv8
│
├── tracking/
│   └── tracker.py               # SORT (Kalman + Hungarian)
│
├── analytics/
│   ├── db.py                    # SQLite event logging
│   ├── analyzer.py              # Usage stats + heatmap
│   └── export.py                # CSV export utility
│
├── prediction/
│   └── predictor.py             # Random Forest demand model
│
├── sterilization/
│   └── simulator.py             # Queue + bottleneck simulation
│
├── dashboard/
│   └── app.py                   # Streamlit live dashboard
│
└── training/
    ├── prepare_dataset.py        # Auto-annotation from synthetic video
    └── train.py                  # YOLOv8 fine-tuning script
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/Eyadi87/surgassist.git
cd surgassist
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Generate sample video**
```bash
python data/generate_sample.py
```

**5. Launch dashboard**
```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🚀 Usage

### Run with Sample Video
1. Launch the dashboard
2. Check **"Use generated sample video"** in the sidebar
3. Click **▶️ Start Processing**
4. Watch live detection, stats, and analytics populate in real time
5. Download the analytics CSV when complete

### Run with Your Own Video
1. Click **"Upload Surgical Video"** in the sidebar
2. Upload any `.mp4`, `.avi`, or `.mov` file
3. Click **▶️ Start Processing**

### Export Analytics
```bash
python analytics/export.py
```
Saves 4 CSV files to `data/exports/`:
- `raw_events.csv` — every detection event
- `usage_summary.csv` — per-instrument breakdown
- `switch_log.csv` — instrument transition log
- `metrics_summary.csv` — OR efficiency metrics

---

## 📈 Metrics Explained

| Metric | Description |
|---|---|
| **OR Efficiency Score** | `100 × (1 - avg_idle × 0.7 - waste_penalty × 0.3)` — higher is better |
| **Waste Estimate %** | % of instruments on tray that were never meaningfully used |
| **Tool Switches** | Number of sustained instrument transitions (debounced, 8-frame minimum) |
| **Idle Ratio** | Fraction of time an instrument was on tray but not in use |
| **Demand Prediction** | Probability distribution over next instrument needed (RF model) |
| **Sterilization Bottleneck** | Queue simulation — flags congestion when demand exceeds capacity |

---

## 🔧 Configuration

All settings in `config.yaml`:

```yaml
instruments:
  - scalpel
  - forceps
  - needle_holder
  - suction

sterilization:
  capacity: 5               # Max instruments in sterilizer at once
  cleaning_duration_min: 45
  turnaround_min: 10

analytics:
  waste_threshold_usage: 5  # Detections below this = wasted instrument

prediction:
  window_size: 10           # Frames used for demand prediction
```

---

## 🎓 Fine-tuning on Real Surgical Data

```bash
# 1. Prepare annotated dataset (YOLO format)
python training/prepare_dataset.py

# 2. Train YOLOv8
python training/train.py

# 3. Update config.yaml with trained model path
# model_path: "models/surgical_instrument_detector/weights/best.pt"
```

Compatible with: **Cholec80**, **CholecT50**, and any YOLO-format surgical dataset.

---

## 🏗 Architecture Decisions

- **Color detection over pure YOLO** for synthetic video — more reliable than mapping COCO classes to surgical instruments
- **SORT over DeepSORT** — lighter weight, runs on CPU without GPU requirement
- **Random Forest over LSTM** — faster training on short sessions, no sequence padding needed
- **SQLite over in-memory** — persistent across page refreshes, exportable
- **Debounced switch counting** — 8-frame minimum prevents noise from counting as switches

---

## 🩺 Clinical Impact

> *"Suction was prepped but never used in this session — removing it from the standard tray could save sterilization cost and preparation time."*

This system gives surgical teams data-driven insight into:
- Which instruments are consistently wasted per procedure type
- When instruments are first needed (optimize tray prep order)
- Peak sterilization load windows (schedule cleaning staff accordingly)
- OR efficiency trends across multiple sessions

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built as a prototype to demonstrate AI-driven surgical workflow optimization. Intended for research and demonstration purposes.*