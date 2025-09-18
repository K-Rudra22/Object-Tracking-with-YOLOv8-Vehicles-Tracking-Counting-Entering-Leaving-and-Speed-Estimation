# 🚗 Object Tracking with YOLOv8: Vehicle Tracking, Counting & Speed Estimation  

This project demonstrates **real-time vehicle tracking, counting, and speed estimation** using **YOLOv8 object detection**, **object tracking**, and **homography-based distance calibration**.  

The system can:  
- 🚙 Detect vehicles in a video stream using YOLOv8  
- 🔄 Track vehicles across frames (with unique IDs)  
- 📊 Count vehicles entering & leaving a scene  
- ⚡ Estimate vehicle speeds (pixels → real-world meters using homography)  
- 📈 Display total vehicles, live counts, and speed overlays  
- 🌐 Run locally or be deployed as a **Streamlit app**  

---

## ✨ Features  
- **YOLOv8 Detection** – State-of-the-art object detection model  
- **Object Tracking** – Keeps IDs consistent for each vehicle  
- **Vehicle Counting** – Tracks entry & exit counts  
- **Speed Estimation** – Uses FPS + homography calibration for approximate real-world speeds  
- **Custom Overlay UI** – Clear badges for counts, IDs, and speeds on video output  
- **Streamlit Deployment** – Run interactively from a browser  

---

## ⚙️ Installation    

Clone the repo and install dependencies:  
```bash
git clone https://github.com/K-Rudra22/Object-Tracking-with-YOLOv8-Vehicles-Tracking-Counting-Entering-Leaving-and-Speed-Estimation.git
cd Object-Tracking-with-YOLOv8-Vehicles-Tracking-Counting-Entering-Leaving-and-Speed-Estimation
pip install -r requirements.txt

---

🎥 Demo


<img width="657" height="332" alt="image" src="https://github.com/user-attachments/assets/f79666ef-be0f-4dad-8db2-5edc1c8a09bd" />




