# Gender and Age Detection Application

This application uses machine learning models to detect the gender and age of individuals in images or videos in real-time. It is built with Streamlit and utilizes OpenCV for face detection.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributions](#contributions)
- [Authors](#authors)

## Features

- Gender detection (Male, Female, Child)
- Age estimation (Child, Adolescent, Young, Adult, Senior)
- Processing of uploaded images
- Real-time detection via webcam

## Installation

Make sure you have Python installed on your machine. Then, install the necessary dependencies:

```bash
pip install streamlit opencv-python pillow numpy scikit-learn cvlib
```

Also, download the pre-trained models (model_svc.pkl and model_svc2.pkl) into the application directory.

## Usage

1.  Clone this repository:

```bash
git clone https://github.com/ayoubboudra1/Age_Gender_detection_using_svm.git
cd Age_Gender_detection_using_svm
```

2.  Run the Streamlit application:

```bash
streamlit run app.py
```

3.  Access the application in your browser at http://localhost:8501.
4.  Choose an image to upload or activate your webcam for real-time detection.

## Authors

- Ayoub BOUDRA
