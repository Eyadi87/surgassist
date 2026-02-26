import yaml
import logging
from collections import deque

logger = logging.getLogger(__name__)


class SterilizationSimulator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        s = cfg["sterilization"]
        self.capacity       = s["capacity"]
        self.clean_duration = s["cleaning_duration_min"]
        self.turnaround     = s["turnaround_min"]
        self.queue          = deque()
        self.in_process     = []

    def set_capacity(self, cap):
        self.capacity = max(1, int(cap))

    def add_instruments(self, used_instruments):
        for inst in used_instruments:
            self.queue.append({"instrument": inst})

    def simulate_cycle(self, time_elapsed_min=60):
        t = 0
        processed        = 0
        congestion_events = []
        timeline         = []

        while t < time_elapsed_min:
            # Finish done items
            done = [x for x in self.in_process if x.get("end", 9999) <= t]
            for d in done:
                self.in_process.remove(d)
                processed += 1
                timeline.append({"time": t, "instrument": d["instrument"], "status": "ready"})

            # Fill slots
            slots = self.capacity - len(self.in_process)
            while self.queue and slots > 0:
                item         = self.queue.popleft()
                item["start"] = t
                item["end"]   = t + self.clean_duration
                self.in_process.append(item)
                slots -= 1

            if len(self.queue) > self.capacity:
                congestion_events.append(t)

            t += self.turnaround

        return {
            "processed":         processed,
            "queue_remaining":   len(self.queue),
            "in_process":        len(self.in_process),
            "congestion_events": len(congestion_events),
            "shortage_risk":     len(congestion_events) > 0,
            "peak_queue":        len(self.queue) + len(self.in_process),
            "timeline":          timeline,
        }

    def get_alert_level(self, result):
        q = result["queue_remaining"]
        if q == 0 and not result["shortage_risk"]:
            return "green",  "No bottleneck detected"
        elif q <= self.capacity:
            return "orange", f"Moderate congestion — {q} instruments queued"
        else:
            return "red",    f"⚠ Shortage risk! {q} instruments pending"