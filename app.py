import streamlit as st
import cv2
import tempfile
import pandas as pd
import numpy as np
import math
from ultralytics import YOLO

# --------------------------
# Load YOLO model
# --------------------------
model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck

st.title("🚦 Vehicle Counting, Speed Tracking")

# --------------------------
# Step 1: Upload video
# --------------------------
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    # --------------------------
    # Step 2: Scaling input
    # --------------------------
    st.markdown("### Enter Pixel-to-Meter Scale")
    st.write("Example: if 50 pixels ≈ 1 meter in your video, enter 50.")
    px_per_meter = st.number_input("Pixels per meter", value=50.0, min_value=1.0)

    if st.button("▶️ Start Processing"):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = "output_final.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        history = {}
        last_positions = {}   # to detect entry/exit
        counted_ids = set()   # to avoid double counting
        rows = []

        vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}
        entering, leaving = 0, 0

        st_frame = st.empty()
        frame_id = 0
        line_y = height // 2  # horizontal line in the middle

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.35)
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Speed in km/h using scale
                    speed_kmph = None
                    prev = history.get(tid, None)
                    if prev is not None:
                        (px, py, pframe) = prev
                        dist_px = math.hypot(cx - px, cy - py)
                        dist_m = dist_px / px_per_meter
                        time_s = max((frame_id - pframe) / fps, 1e-6)
                        speed_kmph = (dist_m / time_s) * 3.6
                        rows.append([int(tid), model.names[int(cls)], frame_id, round(speed_kmph, 2)])

                    history[tid] = (cx, cy, frame_id)

                    # Count unique vehicles
                    if tid not in counted_ids:
                        vehicle_counts[int(cls)] += 1
                        counted_ids.add(tid)

                    # Entry/Exit detection
                    if tid in last_positions:
                        prev_y = last_positions[tid]
                        if prev_y < line_y and cy >= line_y:
                            entering += 1
                        elif prev_y > line_y and cy <= line_y:
                            leaving += 1
                    last_positions[tid] = cy

                    # Draw label
                    label = f"ID:{int(tid)} {model.names[int(cls)]}"
                    if speed_kmph:
                        label += f" {int(speed_kmph)} km/h"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Top bar with counts
            cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 255), -1)
            count_text = " | ".join([f"{model.names[c]}: {vehicle_counts[c]}" for c in vehicle_counts])
            count_text += f" | Entering: {entering} | Leaving: {leaving}"
            cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw the counting line
            cv2.line(frame, (0, line_y), (width, line_y), (255, 255, 0), 2)

            out.write(frame)
            st_frame.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        out.release()

        # ✅ Ensure video file is flushed before showing
        with open(out_path, "rb") as f:
            st.video(f.read())

        # Export CSV
        csv_path = "vehicle_log.csv"
        df = pd.DataFrame(rows, columns=["ID", "Class", "Frame", "Speed_km/h"])
        df.to_csv(csv_path, index=False)

        st.success("✅ Processing complete")
        st.dataframe(df)
        st.download_button("📥 Download CSV", data=open(csv_path, "rb"),
                           file_name="vehicle_log.csv", mime="text/csv")

