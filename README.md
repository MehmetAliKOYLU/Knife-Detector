# Knife Detector

This project is designed to detect and log knives from camera feeds, videos, and images. It was developed to instantly detect knife threats and store the detection in a database for future reference, especially in cases where knives are used in criminal activities.

## Project Overview

The Knife Detector project performs the following key tasks:

1. **Knife Detection**: The system can detect knives in real-time using video feeds, static images, or recorded videos. Once a knife is detected, the system highlights it and logs the detection, including the time and location, into a database.
2. **Custom Dataset Creation**: A custom knife detection dataset was created using various sources. These images were labeled in YOLO format and integrated into the dataset. The dataset was prepared and formatted using Roboflow for easy YOLO model integration.
3. **Model Training**: After assembling the dataset, the model was trained using different algorithms and learning rate values. The best-performing model was selected based on accuracy and performance.
4. **User Interface (GUI)**: The GUI for the project was built using Tkinter, allowing users to interact with the detection system, upload new images or videos for analysis, and view the results of knife detection in real time.
5. **Database Integration**: Detected knife instances, along with the detection time and confidence score, are stored in an SQLite3 database for future retrieval.

## Key Features

- **Real-Time Detection**: Detects knives from live camera feeds.
- **Image & Video Processing**: Processes images and videos to identify knives.
- **Database Logging**: Logs detection time and other relevant details into a database.
- **Adjustable Threshold**: Detection sensitivity threshold can be adjusted based on the use case.

## Setup

To get started with the project, follow these steps:

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/MehmetAliKOYLU/Knife-Detector.git

2. Prepare your dataset in YOLO format or use the provided dataset in the project.

3. Run the project:
   ```bash
   python main.py


## Additional Information

- [YOLO](https://docs.ultralytics.com)
- [Tkínter](https://docs.python.org/3/library/tkinter.html)




