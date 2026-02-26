import yaml
import logging
import sys
from pathlib import Path

def setup_logging(cfg):
    log_cfg  = cfg.get("logging", {})
    level    = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file", "logs/surgassist.log")
    Path(log_file).parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ]
    )

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    setup_logging(cfg)
    logging.info("SurgAssist ready.")
    logging.info("Step 1: python data/generate_sample.py")
    logging.info("Step 2: streamlit run dashboard/app.py")