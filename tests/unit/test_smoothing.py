import numpy as np

from inference.smoothing.temporal import PredictionSmoother


def test_prediction_smoother_stabilizes():
    s = PredictionSmoother(alpha=0.7, confirm_frames=2, vote_window=4)
    probs_a = np.array([0.8, 0.2], dtype=np.float32)
    probs_b = np.array([0.2, 0.8], dtype=np.float32)

    lbl, _ = s.update(probs_a)
    assert lbl in (0, 1)
    lbl, _ = s.update(probs_a)
    assert lbl == 0
    lbl, _ = s.update(probs_b)
    lbl, _ = s.update(probs_b)
    assert lbl in (0, 1)
