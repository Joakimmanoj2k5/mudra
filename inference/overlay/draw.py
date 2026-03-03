from typing import Dict

import cv2


def draw_overlay(frame, result: Dict[str, object], fps: float, target: str = ""):
    status = result.get("status", "")
    if status == "ok":
        color = (52, 211, 153)  # green
    elif status == "uncertain":
        color = (45, 212, 245)  # yellow-ish cyan in BGR
    else:
        color = (248, 113, 113)  # red
    label = result.get("label", "-")
    conf = result.get("confidence", 0.0)
    model = result.get("model_used", "-")
    env = result.get("env", {})
    perf_warn = result.get("perf_warning", "")

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (520, 195), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"Target: {target or '-'}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (203, 213, 225), 2)
    cv2.putText(frame, f"Model: {model}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (148, 163, 184), 2)
    cv2.putText(frame, f"Pred: {label} ({conf:.2f})", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"Status: {status}", (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (148, 163, 184), 2)
    if perf_warn:
        cv2.putText(frame, perf_warn, (20, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 113, 113), 2)

    env_txt = (
        f"Environment: MediaPipe: {'OK' if env.get('mediapipe') else 'X'}  "
        f"Torch: {'OK' if env.get('torch') else 'X'}  "
        f"Camera: {'OK' if env.get('camera') else 'X'}"
    )
    cv2.putText(frame, env_txt, (20, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (203, 213, 225), 1)
    return frame
