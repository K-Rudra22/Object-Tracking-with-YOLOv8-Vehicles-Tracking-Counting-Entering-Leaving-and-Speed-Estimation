Overview
This project demonstrates real-time vehicle tracking, counting, and speed estimation using YOLOv8 object detection combined with object tracking and homography-based distance calibration.
The system can:
•	Detect vehicles in a video stream using YOLOv8.
•	Track vehicles across frames (with unique IDs).
•	 Count vehicles entering & leaving a scene.
•	 Estimate vehicle speeds (pixels → real-world meters using homography).
•	 Display total vehicles, live counts, and speed overlays with improved visibility.
•	 Run locally or be deployed as a Streamlit app.
________________________________________
🎥 Demo


<img width="657" height="332" alt="image" src="https://github.com/user-attachments/assets/f79666ef-be0f-4dad-8db2-5edc1c8a09bd" />


________________________________________
⚙️ Features
•	YOLOv8 Detection – State-of-the-art object detection model.
•	Object Tracking – Keeps IDs consistent for each vehicle.
•	Vehicle Counting – Tracks entry & exit counts.
•	Speed Estimation – Uses FPS + homography calibration for approximate real-world speeds.
•	Custom Overlay UI – Clear badges for counts, IDs, and speeds on video output.
•	Streamlit Deployment – Run interactively from a browser.
