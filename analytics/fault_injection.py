import random
import yaml
import logging

logger = logging.getLogger(__name__)

class FaultInjector:
    """
    Simulates OR failure scenarios:
    - Instrument mislabel (sensor drift)
    - Sudden instrument dropout (occlusion)
    - Sterilizer failure (capacity drop)
    - False detection spike
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.instruments = self.cfg["instruments"]
        self.active_faults = []

    def inject_mislabel(self, detections, probability=0.1):
        for det in detections:
            if random.random() < probability:
                original = det["label"]
                det["label"] = random.choice(self.instruments)
                logger.warning(f"[FAULT] Mislabel: {original} → {det['label']}")
        return detections

    def inject_dropout(self, detections, probability=0.15):
        surviving = []
        for det in detections:
            if random.random() > probability:
                surviving.append(det)
            else:
                logger.warning(f"[FAULT] Dropout: {det['label']} lost")
        return surviving

    def inject_false_detection(self, detections, frame_shape=(480, 640)):
        if random.random() < 0.08:
            fake_label = random.choice(self.instruments)
            h, w = frame_shape[:2]
            x1 = random.randint(0, w - 80)
            y1 = random.randint(0, h - 40)
            detections.append({
                "bbox": [x1, y1, x1 + 80, y1 + 40],
                "confidence": round(random.uniform(0.3, 0.5), 2),
                "label": fake_label,
                "cls_id": -1,
                "_is_fake": True
            })
            logger.warning(f"[FAULT] False detection injected: {fake_label}")
        return detections

    def inject_sterilizer_failure(self, simulator):
        original = simulator.capacity
        simulator.capacity = max(1, simulator.capacity // 2)
        self.active_faults.append(("sterilizer_capacity", original))
        logger.warning(f"[FAULT] Sterilizer capacity halved: {original} → {simulator.capacity}")
        return simulator

    def restore_all(self, simulator):
        for fault_type, original_val in self.active_faults:
            if fault_type == "sterilizer_capacity":
                simulator.capacity = original_val
                logger.info(f"[RESTORE] Sterilizer capacity restored to {original_val}")
        self.active_faults.clear()
        return simulator

    def run_scenario(self, scenario_name, detections, simulator=None, frame_shape=(480, 640)):
        scenarios = {
            "sensor_drift": lambda d: self.inject_mislabel(d, 0.2),
            "occlusion_burst": lambda d: self.inject_dropout(d, 0.4),
            "false_alarm": lambda d: self.inject_false_detection(d, frame_shape),
            "sterilizer_down": lambda d: d,
        }
        if scenario_name not in scenarios:
            logger.error(f"Unknown fault scenario: {scenario_name}")
            return detections, simulator
        detections = scenarios[scenario_name](detections)
        if scenario_name == "sterilizer_down" and simulator:
            simulator = self.inject_sterilizer_failure(simulator)
        logger.info(f"[FAULT SCENARIO] '{scenario_name}' applied.")
        return detections, simulator