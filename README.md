# Webcam Object Detection :

This project uses YOLOv5x (a powerful deep learning object detection model) to detect and label objects in real-time using your webcam.

📷 Demo
Real-time detection of objects like person, laptop, phone, cup, bottle, chair, etc. using YOLOv5x.

🚀 Requirements
Python 3.7+

Internet connection (for first-time model download)

🔧 Install dependencies
bash
Copy
Edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy
💡 If you have a GPU, install the appropriate CUDA version of PyTorch from https://pytorch.org/get-started/locally/

▶️ How to Run
bash
Copy
Edit
python detect_webcam.py
Press q to quit the detection window.

📄 Script Overview (detect_webcam.py)
Loads YOLOv5x from PyTorch Hub

Sets detection confidence threshold to 0.5

Opens webcam and processes each frame

Draws bounding boxes and labels detected objects

🧠 Model Used
YOLOv5x – The most accurate model in the YOLOv5 family

Trained on the COCO dataset (detects 80+ common object classes)

🎯 Customize Detection
To detect only specific objects, modify this line:

python
Copy
Edit
model.classes = [0]  # Example: Only detect persons (class ID 0)
You can find all class IDs here

📝 Notes
Ensure your webcam is accessible

Best results under good lighting conditions

Resize the input frame if needed for speed or quality
