

---

# Real-Time Number Plate Detection using YOLOv7

## Project Overview

This project implements a **Real-Time Number Plate Detection System** using **YOLOv7**, a state-of-the-art object detection algorithm. The system is designed to identify and recognize vehicle license plates from live video streams or images captured by surveillance cameras in urban environments. By leveraging deep learning techniques, this project aims to enhance traffic monitoring, law enforcement, and security systems, with applications in **traffic management**, **toll collection**, **automated parking systems**, and more.

### Features
- Real-time license plate detection using YOLOv7
- High detection accuracy via fine-tuning on a custom dataset
- GPU support with CUDA for accelerated performance
- Easily retrainable and customizable for specific needs

---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Real-Time Detection](#real-time-detection)
- [Evaluation](#evaluation)
- [Technologies](#technologies)
- [License](#license)

---

## Installation

Follow these steps to set up the project on your local machine or any server:

1. **Clone the repository**:
   Open your terminal and run:
   ```bash
   git clone https://github.com/username/Real-Time-Number-Plate-Detection.git
   ```

2. **Navigate into the project directory**:
   ```bash
   cd Real-Time-Number-Plate-Detection
   ```

3. **Set up a Python environment**:
   It's highly recommended to use a virtual environment to manage dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   env\Scripts\activate     # For Windows
   ```

4. **Install required dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   This will install necessary libraries like `torch`, `opencv-python`, `matplotlib`, and others.

5. **Download Pre-trained YOLOv7 Weights**:
   You can use the pre-trained YOLOv7 weights, or you can train your own model (explained later). To download the pre-trained weights:
   - Go to the official YOLOv7 repository or another trusted source for pre-trained weights.
   - Place the weights file (e.g., `yolov7.pt`) in the `models/` directory.

---

## Usage

Once the environment is set up, you can use the system for either **training** the model or **detecting license plates** in real-time.

### Running Real-Time Detection

To use the trained YOLOv7 model to detect number plates in a live video stream or an image, run the `detect.py` script.

1. **Run detection on a live video stream**:
   ```bash
   python detect.py --source 0  # 0 represents the webcam
   ```

2. **Run detection on a video file**:
   ```bash
   python detect.py --source path_to_video_file.mp4
   ```

3. **Run detection on an image**:
   ```bash
   python detect.py --source path_to_image.jpg
   ```

The model will process the video/image and display the number plate detections in real-time, drawing bounding boxes around detected plates.

---

## Project Structure

Here is a breakdown of the project directory:

```plaintext
Real-Time-Number-Plate-Detection/
│
├── data/                     # Training and test dataset with annotated images
│
├── models/                   # Contains YOLOv7 config and weights files
│   └── yolov7.pt             # Pre-trained YOLOv7 model
│
├── scripts/                  # Python scripts for training and detection
│   ├── train.py              # Script to train the YOLOv7 model
│   ├── detect.py             # Real-time detection script
│   └── evaluate.py           # Evaluation script
│
├── requirements.txt          # List of required Python libraries
└── README.md                 # Project documentation
```

---

## Training the Model

If you want to train the model on your own dataset (for example, a new set of license plates), you need to follow these steps:

1. **Prepare the dataset**:  
   Collect and annotate images of vehicles with visible license plates. Use annotation tools like [LabelImg](https://github.com/tzutalin/labelImg) to label the bounding boxes around the plates.
   
2. **Preprocess the data**:  
   Organize the dataset into appropriate folders (`train/`, `test/`) and ensure that each image has an associated `.txt` file containing the bounding box coordinates and class labels.

3. **Configure YOLOv7**:
   - Create a custom dataset configuration file (based on `data.yaml`).
   - Update the `train.py` script to load your custom dataset.

4. **Train the model**:  
   Run the following command to begin training:
   ```bash
   python train.py --data path_to_data.yaml --cfg yolov7.yaml --weights yolov7.pt --batch-size 16 --epochs 50
   ```

   This command trains the YOLOv7 model using the pre-trained weights and fine-tunes it for your custom dataset.

---

## Real-Time Detection

After training the model, you can use the `detect.py` script to run the trained model in real-time for license plate detection:

1. **Run detection on a live video stream (webcam)**:
   ```bash
   python detect.py --source 0
   ```

2. **Run detection on a video file**:
   ```bash
   python detect.py --source path_to_video.mp4
   ```

3. **Run detection on a single image**:
   ```bash
   python detect.py --source path_to_image.jpg
   ```

The system will draw bounding boxes around detected license plates and display them in real-time.

---

## Evaluation

To evaluate the model performance, you can use the `evaluate.py` script, which calculates key metrics such as **Precision**, **Recall**, and **mAP** (mean Average Precision).

Run the following command to evaluate the model on a test dataset:

```bash
python evaluate.py --weights path_to_trained_weights.pt --data path_to_data.yaml
```

---

## Technologies

- **YOLOv7**: A state-of-the-art object detection algorithm used to detect license plates in real-time.
- **Python**: The programming language used to implement the project.
- **OpenCV**: For image and video processing.
- **PyTorch or TensorFlow**: Deep learning frameworks used for training and inference.
- **CUDA**: For GPU acceleration to speed up model inference.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Conclusion

This project allows you to implement real-time license plate detection using YOLOv7 and offers flexibility to train the model on your own dataset for custom applications. Whether for traffic management, law enforcement, or parking systems, this system provides an efficient and scalable solution for vehicle identification.

---


