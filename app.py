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

st.title("🚦 Vehicle Counting & Speed Tracking (No Homography)")

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

    if st.button("▶️ Start Processing"):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = "output_simple.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        history = {}
        rows = []
        vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}
        st_frame = st.empty()
        frame_id = 0

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

                    # Pixel-based speed (no homography)
                    speed = None
                    prev = history.get(tid, None)
                    if prev is not None:
                        (px, py, pframe) = prev
                        dist_px = math.hypot(cx - px, cy - py)
                        time_s = max((frame_id - pframe) / fps, 1e-6)
                        speed = dist_px / time_s  # pixels per second
                        rows.append([int(tid), model.names[int(cls)], frame_id, round(speed, 2)])

                    history[tid] = (cx, cy, frame_id)
                    vehicle_counts[int(cls)] += 1

                    # Draw label
                    label = f"ID:{int(tid)} {model.names[int(cls)]}"
                    if speed:
                        label += f" {int(speed)} px/s"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Top bar with counts
            cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 255), -1)
            count_text = " | ".join([f"{model.names[c]}: {vehicle_counts[c]}" for c in vehicle_counts])
            cv2.putText(frame, count_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)
            st_frame.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        out.release()

        # Show results
        st.success("✅ Processing complete")
        st.video(out_path)

        # Export CSV
        csv_path = "vehicle_log.csv"
        df = pd.DataFrame(rows, columns=["ID", "Class", "Frame", "Speed_px/s"])
        df.to_csv(csv_path, index=False)

        st.dataframe(df)
        st.download_button("📥 Download CSV", data=open(csv_path, "rb"),
                           file_name="vehicle_log.csv", mime="text/csv")
