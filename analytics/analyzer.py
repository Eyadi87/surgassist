import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.max_open_warning"] = 50
import yaml
import logging

logger = logging.getLogger(__name__)


class UsageAnalyzer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.fps             = 25
        self.waste_threshold = cfg["analytics"]["waste_threshold_usage"]
        self.instruments     = cfg["instruments"]

    def compute_stats(self, df):
        stats = {}
        for label in self.instruments:
            if df.empty:
                stats[label] = {
                    "total_frames": 0, "usage_sec": 0.0,
                    "first_seen": None, "is_waste": True, "idle_ratio": 1.0
                }
                continue
            sub = df[df["label"] == label]
            if sub.empty:
                stats[label] = {
                    "total_frames": 0, "usage_sec": 0.0,
                    "first_seen": None, "is_waste": True, "idle_ratio": 1.0
                }
                continue

            total      = len(sub)
            min_frame  = int(sub["frame_id"].min())
            max_frame  = int(sub["frame_id"].max())
            frame_span = max(max_frame - min_frame + 1, 1)
            idle       = max(0, frame_span - total)

            stats[label] = {
                "total_frames": total,
                "usage_sec":    round(total / self.fps, 2),
                "first_seen":   min_frame,
                "is_waste":     total < self.waste_threshold,
                "idle_ratio":   round(idle / frame_span, 3),
            }
        return stats

    def compute_switch_sequence(self, df):
        if df.empty:
            return []
        # Get dominant instrument per frame
        dominant = (
            df.groupby("frame_id")["label"]
            .agg(lambda x: x.value_counts().index[0])
            .sort_index()
            .values
        )
        # Debounce: only count switch if new instrument persists 8+ consecutive frames
        DEBOUNCE = 8
        switches = []
        i = 0
        while i < len(dominant) - DEBOUNCE:
            if dominant[i] != dominant[i + 1]:
                new_inst = dominant[i + 1]
                window   = dominant[i + 1 : i + 1 + DEBOUNCE]
                if all(w == new_inst for w in window):
                    switches.append((dominant[i], new_inst))
                    i += DEBOUNCE
                    continue
            i += 1
        return switches

    def waste_percentage(self, stats):
        if not stats:
            return 0.0
        waste = sum(1 for v in stats.values() if v["is_waste"])
        return round(waste / len(stats) * 100, 1)

    def or_efficiency_score(self, stats):
        if not stats:
            return 0.0
        active    = [v for v in stats.values() if not v["is_waste"]]
        if not active:
            return 0.0
        avg_idle  = np.mean([v["idle_ratio"] for v in active])
        waste_pen = sum(1 for v in stats.values() if v["is_waste"]) / len(stats)
        score     = max(0.0, 100.0 * (1.0 - avg_idle * 0.7 - waste_pen * 0.3))
        return round(score, 1)

    def heatmap_figure(self, df):
        plt.close("all")   # prevent memory leak
        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        if df.empty:
            ax.text(0.5, 0.5, "No data yet", color="white",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        try:
            # Bucket frames into 100 bins for readability
            df2              = df.copy()
            df2["frame_bin"] = pd.cut(df2["frame_id"], bins=100, labels=False)
            pivot = (
                df2.groupby(["label", "frame_bin"])
                   .size()
                   .unstack(fill_value=0)
            )
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, color="white", fontsize=10)
            ax.set_xlabel("Video progress →", color="#aaaaaa", fontsize=9)
            ax.tick_params(colors="white")
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        except Exception as e:
            logger.warning(f"Heatmap error: {e}")
            ax.text(0.5, 0.5, "Generating...", color="white",
                    ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        return fig