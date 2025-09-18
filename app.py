import streamlit as st
import cv2
import tempfile
import pandas as pd
import numpy as np
import math
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --------------------------
# Load YOLO model
# --------------------------
model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck

st.title("🚦 Vehicle Counting & Speed Tracking (Homography)")

# --------------------------
# Step 1: Upload video
# --------------------------
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        st.error("❌ Could not read video file. Please upload a valid video.")
    else:
        st.markdown("### Step 2: Pick 4 points on the road (top-left, top-right, bottom-left, bottom-right)")
        st.image(first_frame, caption="Click 4 points in order")

        # Convert frame for canvas (OpenCV -> PIL)
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame_pil = Image.fromarray(first_frame_rgb)

        # Interactive canvas
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)", stroke_width=3, stroke_color="red",
            background_image=first_frame_pil, update_streamlit=True,
            height=first_frame.shape[0], width=first_frame.shape[1],
            drawing_mode="point", key="canvas"
        )

        # Extract calibration points
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) == 4:
                st.success("✅ 4 points selected")
                image_points = np.array([[obj["left"], obj["top"]] for obj in objects], dtype=np.float32)

                # Ask user for real-world rectangle dimensions
                st.markdown("### Step 3: Enter real-world rectangle dimensions (in meters)")
                width_m = st.number_input("Width (m)", value=10.0)
                height_m = st.number_input("Height (m)", value=20.0)

                world_points = np.array([
                    [0, 0],
                    [width_m, 0],
                    [0, height_m],
                    [width_m, height_m]
                ], dtype=np.float32)

                H, _ = cv2.findHomography(image_points, world_points)
                st.session_state["H"] = H
                st.success("Homography matrix computed ✅")

# --------------------------
# Step 4: Process video
# --------------------------
if uploaded_file is not None and "H" in st.session_state:
    if st.button("▶️ Start Processing"):
        H = st.session_state["H"]

        def project_to_world(pt, H):
            px = np.array([pt[0], pt[1], 1.0])
            wx = H @ px
            wx /= wx[2]
            return wx[0], wx[1]

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = "output_homography.mp4"
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

                    # Homography projection
                    X, Y = project_to_world((cx, cy), H)

                    speed_kmph = None
                    prev = history.get(tid, None)
                    if prev is not None:
                        (pX, pY, pframe) = prev
                        dist_m = math.hypot(X - pX, Y - pY)
                        time_s = max((frame_id - pframe) / fps, 1e-6)
                        speed_kmph = (dist_m / time_s) * 3.6
                        rows.append([int(tid), model.names[int(cls)], frame_id, round(speed_kmph, 2)])

                    history[tid] = (X, Y, frame_id)
                    vehicle_counts[int(cls)] += 1

                    # Draw labels
                    label = f"ID:{int(tid)} {model.names[int(cls)]}"
                    if speed_kmph:
                        label += f" {int(speed_kmph)} km/h"

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
        df = pd.DataFrame(rows, columns=["ID", "Class", "Frame", "Speed_km/h"])
        df.to_csv(csv_path, index=False)

        st.dataframe(df)
        st.download_button("📥 Download CSV", data=open(csv_path, "rb"),
                           file_name="vehicle_log.csv", mime="text/csv")
