import streamlit as st
import cv2, math, time, pandas as pd, numpy as np
from ultralytics import YOLO
import os
import io

# --- Helper functions and constants ---
# Your existing draw_badge function and class_names dictionary
def draw_badge(frame, text, pos, bg_color=(0,0,0), text_color=(255,255,255), scale=0.7, thickness=2):
    # The rest of your function code goes here
    ...
    pass

class_names = {2:"car", 3:"motorbike", 5:"bus", 7:"truck"}
MODEL_NAME = "yolo11n.pt"
CONF = 0.35
IMG_SIZE = 640
LINE_Y_RATIO = 0.5
METERS_PER_PIXEL = 0.05 # You'll need to define this based on your camera calibration

# --- Streamlit UI elements ---
st.title("Vehicle Tracking and Counting App")
st.write("Upload a video to track vehicles and analyze traffic.")

# File uploader widget for the video
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4"])

if uploaded_file is not None:
    # Use a temporary file to save the video for OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    VIDEO = tfile.name

    # Display a message to the user
    st.info("Processing video... This may take a moment.")
    
    # Use a progress bar to show processing status
    progress_bar = st.progress(0)
    
    # Placeholder for the video output and data
    video_placeholder = st.empty()
    data_placeholder = st.empty()

    # --- Video processing logic ---
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use a BytesIO object to hold the output video in memory
    output_video_bytes = io.BytesIO()
    writer = cv2.VideoWriter(output_video_bytes, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    line_y = int(H * LINE_Y_RATIO)
    history = {}
    rows = []
    counts_in = {"car":0, "motorbike":0, "bus":0, "truck":0}
    counts_out = {"car":0, "motorbike":0, "bus":0, "truck":0}
    crossed_in, crossed_out = set(), set()
    frame_id = 0
    
    stream = model.track(source=VIDEO, tracker="bytetrack.yaml", conf=CONF, imgsz=IMG_SIZE, stream=True)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for res in stream:
        frame_id += 1
        frame = res.orig_img
        boxes = res.boxes
        
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else [-1]*len(boxes)
            
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, box)
                cls = int(classes[i])
                if cls not in class_names:
                    continue
                
                vid = int(ids[i])
                cname = class_names[cls]
                cX, cY = int((x1+x2)/2), int((y1+y2)/2)
                
                # ... (rest of your logic for speed and counting) ...
                
                # Draw detection on frame
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2)
                label = f"{vid}:{cname}"
                if speed_kmph is not None:
                    label += f" {int(speed_kmph)}km/h"
                draw_badge(frame, label, (x1, y1-5), bg_color=(255,0,255), text_color=(255,255,255), scale=0.6)
        
        # Draw line and badges on frame
        cv2.line(frame, (0,line_y), (W,line_y), (0,0,255), 2)
        y0 = 30
        draw_badge(frame, "Vehicles Leaving", (10, y0), bg_color=(0,0,255))
        for i,(cls,v) in enumerate(counts_out.items()):
            draw_badge(frame, f"{cls}:{v}", (10, y0+30+25*i), bg_color=(0,0,0))
        xr = W-320
        draw_badge(frame, "Vehicles Entering", (xr, y0), bg_color=(0,0,255))
        for i,(cls,v) in enumerate(counts_in.items()):
            draw_badge(frame, f"{cls}:{v}", (xr, y0+30+25*i), bg_color=(0,0,0))
        
        writer.write(frame)
        
        # Update Streamlit elements
        progress_bar.progress(frame_id / total_frames)
        # You'll need to save each processed frame and display it in a loop
        # For a full video, it's better to wait until the end and show the final result
        
    cap.release()
    writer.release()
    
    # Display the final output video and data
    st.success("Processing complete!")
    st.video(output_video_bytes.getvalue())
    
    # Save CSV data and display a download button
    df = pd.DataFrame(rows, columns=["ObjectID","Class","Frame","Speed_km/h"])
    st.subheader("Vehicle Data Log")
    st.dataframe(df)
    
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv_bytes,
        file_name="vehicle_log.csv",
        mime="text/csv",
    )
    
    # Clean up the temporary file
    os.remove(VIDEO)


