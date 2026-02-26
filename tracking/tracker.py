import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w   = max(0., xx2 - xx1)
    h   = max(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    return inter / float(area1 + area2 - inter + 1e-6)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, label):
        self.kf        = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F      = np.eye(7)
        self.kf.F[0,4] = self.kf.F[1,5] = self.kf.F[2,6] = 1
        self.kf.H      = np.zeros((4, 7))
        np.fill_diagonal(self.kf.H, 1)
        self.kf.R     *= 10
        self.kf.P[4:, 4:] *= 1000
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        x1, y1, x2, y2 = bbox
        self.kf.x[:4]  = np.array([x1, y1, x2, y2]).reshape(4, 1)

        self.id            = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.label         = label
        self.hits          = 1
        self.no_losses     = 0
        self.active_frames = 1

    def update(self, bbox):
        self.kf.update(np.array(bbox).reshape(4, 1))
        self.hits         += 1
        self.no_losses     = 0
        self.active_frames += 1

    def predict(self):
        self.kf.predict()
        self.no_losses += 1
        return self.kf.x[:4].flatten().tolist()

    def get_state(self):
        return self.kf.x[:4].flatten().tolist()


class SORT:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.2):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []

    def update(self, detections):
        # Predict all existing trackers
        for t in self.trackers:
            t.predict()

        matched        = []
        unmatched_dets = list(range(len(detections)))

        if self.trackers and detections:
            iou_matrix = np.zeros((len(detections), len(self.trackers)))
            for d, det in enumerate(detections):
                for t, trk in enumerate(self.trackers):
                    iou_matrix[d, t] = iou(det["bbox"], trk.get_state())

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched.append((r, c))
                    unmatched_dets.remove(r)

        # Update matched
        for d, t in matched:
            self.trackers[t].update(detections[d]["bbox"])
            self.trackers[t].label = detections[d]["label"]

        # Create new trackers
        for d in unmatched_dets:
            self.trackers.append(
                KalmanBoxTracker(detections[d]["bbox"], detections[d]["label"])
            )

        # Return active trackers
        active = []
        for t in self.trackers:
            if t.no_losses <= self.max_age and t.hits >= self.min_hits:
                active.append({
                    "id":            t.id,
                    "bbox":          t.get_state(),
                    "label":         t.label,
                    "active_frames": t.active_frames,
                })

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]
        return active