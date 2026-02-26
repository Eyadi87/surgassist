import streamlit as st
import cv2
import yaml
import pandas as pd
import numpy as np
import sys
import os
import logging
import tempfile
import subprocess
from pathlib import Path

# ── Fix imports ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Auto-generate sample video if missing ─────────────────────────────────────
video_path = os.path.join(ROOT, "data", "videos", "sample.mp4")
if not os.path.exists(video_path):
    os.makedirs(os.path.join(ROOT, "data", "videos"), exist_ok=True)
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "data", "generate_sample.py")],
        check=True,
        cwd=ROOT
    )

from detection.detector      import SurgicalDetector
from tracking.tracker        import SORT
from analytics.db            import init_db, log_event, get_events_df, clear_session
from analytics.analyzer      import UsageAnalyzer
from prediction.predictor    import InstrumentPredictor
from sterilization.simulator import SterilizationSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG  = os.path.join(ROOT, "config.yaml")
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

DB_PATH = os.path.join(ROOT, cfg["output_db"])
init_db(DB_PATH)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SurgAssist", layout="wide", page_icon="🏥")
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1117; }
.stMetric label { color: #8899aa !important; font-size: 12px; }
.stMetric div   { color: #e0e6f0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 SurgAssist — Surgical Instrument Intelligence")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("**Tracked Instruments:**")
    for i in cfg["instruments"]:
        st.write(f"  • {i}")

    st.divider()
    steril_cap  = st.slider("Sterilization Capacity",  1, 20, cfg["sterilization"]["capacity"])
    conf_thresh = st.slider("Detection Confidence", 0.1, 0.9, cfg["detection"]["confidence"], 0.05)

    st.divider()
    uploaded   = st.file_uploader("Upload Surgical Video", type=["mp4", "avi", "mov"])
    use_sample = st.checkbox("Use generated sample video", value=True)

    st.divider()
    if st.button("🗑 Clear Session Data"):
        clear_session(DB_PATH)
        st.success("Session cleared.")

# ── Main layout ───────────────────────────────────────────────────────────────
col_video, col_stats = st.columns([2, 1])
with col_video:
    st.subheader("📹 Live Detection Feed")
    frame_ph = st.empty()
with col_stats:
    st.subheader("📊 Real-Time Stats")
    stats_ph = st.empty()

st.divider()
col_pred, col_steril, col_metrics = st.columns(3)
with col_pred:
    st.subheader("🔮 Demand Prediction")
    pred_ph = st.empty()
with col_steril:
    st.subheader("🧪 Sterilization Bottleneck")
    steril_ph = st.empty()
with col_metrics:
    st.subheader("📈 OR Efficiency Metrics")
    metrics_ph = st.empty()

st.divider()
st.subheader("🗺 Instrument Usage Heatmap")
heatmap_ph = st.empty()


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline():
    detector  = SurgicalDetector(CONFIG)
    detector.conf = conf_thresh
    tracker   = SORT(max_age=30, min_hits=1, iou_threshold=0.2)
    analyzer  = UsageAnalyzer(CONFIG)
    predictor = InstrumentPredictor(CONFIG)
    predictor.load()
    simulator = SterilizationSimulator(CONFIG)
    simulator.set_capacity(steril_cap)

    # ── Video source ──
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        source = tfile.name
    else:
        source = os.path.join(ROOT, cfg["video_source"])

    if not Path(source).exists():
        st.error(
            f"Video not found at `{source}`.\n\n"
            "Run this first:  `python data/generate_sample.py`"
        )
        return

    cap          = cv2.VideoCapture(source)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id     = 0
    recent_labels = []
    RETRAIN_EVERY = 150

    progress_bar = st.progress(0, text="Starting…")
    stop_ph      = st.empty()
    stop_btn     = stop_ph.button("⏹ Stop Processing")

    COLORS = {
        "scalpel":       (50,  50,  220),
        "forceps":       (50,  200, 50),
        "needle_holder": (220, 50,  50),
        "suction":       (50,  200, 220),
    }

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp  = frame_id / fps
        detections = detector.detect(frame)
        tracked    = tracker.update(detections)

        # ── Persist to DB ──
        for t in tracked:
            log_event(DB_PATH, timestamp, frame_id,
                      t["id"], t["label"], t["bbox"], t["active_frames"])
            recent_labels.append(t["label"])

        # ── Draw annotations ──
        annotated = frame.copy()
        for t in tracked:
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            color = COLORS.get(t["label"], (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Instrument name ABOVE box
            cv2.putText(annotated,
                        t["label"],
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            # Track ID BELOW box
            cv2.putText(annotated,
                        f"ID:{t['id']}",
                        (x1, min(y2 + 14, annotated.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 100), 1)

        # Clean timestamp with black background
        ts_text = f"  {timestamp:.1f}s  |  Frame {frame_id}  "
        cv2.rectangle(annotated,
                      (0, annotated.shape[0] - 22),
                      (320, annotated.shape[0]),
                      (0, 0, 0), -1)
        cv2.putText(annotated, ts_text,
                    (8, annotated.shape[0] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                       channels="RGB", width=700)

        # ── Analytics update every 25 frames ──
        if frame_id % 25 == 0 and frame_id > 0:
            df    = get_events_df(DB_PATH)
            stats = analyzer.compute_stats(df)

            # Real-Time Stats
            with stats_ph.container():
                for inst, s in stats.items():
                    icon   = "🟢" if not s["is_waste"] else "🔴"
                    filled = int((1 - s["idle_ratio"]) * 20)
                    bar    = "█" * filled + "░" * (20 - filled)
                    st.markdown(
                        f"{icon} **{inst}**  \n"
                        f"`{bar}` **{s['usage_sec']}s** | Idle: {int(s['idle_ratio']*100)}%"
                    )

            # Retrain predictor
            if frame_id % RETRAIN_EVERY == 0 and len(df) > 30:
                predictor.train(df)

            # Demand prediction
            pred = predictor.predict_next(
                recent_labels[-cfg["prediction"]["window_size"]:]
            )
            with pred_ph.container():
                for inst, prob in sorted(pred.items(), key=lambda x: -x[1]):
                    st.progress(float(prob), text=f"{inst}: {int(prob*100)}%")

            # Sterilization
            used  = [i for i, s in stats.items() if not s["is_waste"]]
            simulator.add_instruments(used)
            sim   = simulator.simulate_cycle()
            level, msg = simulator.get_alert_level(sim)
            emoji = {"green": "🟢", "orange": "🟡", "red": "🔴"}[level]
            with steril_ph.container():
                st.markdown(f"**{emoji} {msg}**")
                a, b, c = st.columns(3)
                a.metric("Processed",  sim["processed"])
                b.metric("In Queue",   sim["queue_remaining"])
                c.metric("Congestion", sim["congestion_events"])

            # OR Efficiency
            waste_pct = analyzer.waste_percentage(stats)
            or_score  = analyzer.or_efficiency_score(stats)
            switches  = analyzer.compute_switch_sequence(df)
            active_n  = len([s for s in stats.values() if not s["is_waste"]])
            with metrics_ph.container():
                a, b = st.columns(2)
                a.metric("OR Efficiency",    f"{or_score}/100")
                b.metric("Waste Estimate",   f"{waste_pct}%")
                c, d = st.columns(2)
                c.metric("Tool Switches",    len(switches))
                d.metric("Active Instruments", active_n)

            # Heatmap
            if not df.empty:
                heatmap_ph.pyplot(analyzer.heatmap_figure(df), clear_figure=True)

        # Progress bar
        if total_frames > 0:
            pct = min(frame_id / total_frames, 1.0)
            progress_bar.progress(pct, text=f"Frame {frame_id}/{total_frames} — {timestamp:.1f}s")

        frame_id += 1

    cap.release()
    if uploaded:
        try:
            os.unlink(tfile.name)
        except Exception:
            pass

    progress_bar.progress(1.0, text="✅ Done!")
    stop_ph.empty()

    # ── Final report ──────────────────────────────────────────────────────────
    df_final = get_events_df(DB_PATH)
    if df_final.empty:
        st.warning("⚠ No tracking data recorded. The video may not have loaded correctly.")
        return

    st.divider()
    st.subheader("📋 Final Session Report")

    stats_f    = analyzer.compute_stats(df_final)
    waste_f    = analyzer.waste_percentage(stats_f)
    or_f       = analyzer.or_efficiency_score(stats_f)
    switches_f = analyzer.compute_switch_sequence(df_final)
    active_f   = len([s for s in stats_f.values() if not s["is_waste"]])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("OR Efficiency",      f"{or_f}/100")
    c2.metric("Waste Estimate",     f"{waste_f}%")
    c3.metric("Tool Switches",      len(switches_f))
    c4.metric("Active Instruments", active_f)
    c5.metric("Total Frames",       int(df_final["frame_id"].max()))

    st.subheader("📊 Per-Instrument Breakdown")
    rows = []
    for inst, s in stats_f.items():
        rows.append({
            "Instrument":         inst,
            "Usage (sec)":        s["usage_sec"],
            "Detections":         s["total_frames"],
            "Idle %":             f"{int(s['idle_ratio']*100)}%",
            "Waste Flag":         "⚠ Wasted" if s["is_waste"] else "✅ Used",
            "First Seen (frame)": int(s["first_seen"]) if s["first_seen"] is not None else 0,
        })
    st.dataframe(pd.DataFrame(rows), width=1400)

    # Final heatmap
    st.subheader("🗺 Final Usage Heatmap")
    import matplotlib.pyplot as plt
    fig_final = analyzer.heatmap_figure(df_final)
    st.pyplot(fig_final, clear_figure=True)
    plt.close("all")

    # CSV download
    csv = df_final.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Analytics CSV",
        data=csv,
        file_name="surgassist_report.csv",
        mime="text/csv",
    )


if st.button("▶️ Start Processing", type="primary"):
    run_pipeline()