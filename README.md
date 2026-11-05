# SIFT Face Recognition Studio

A beginner-friendly Python desktop project that demonstrates **face identification using SIFT (Scale-Invariant Feature Transform)**, Euclidean-distance descriptor matching, and a simple Tkinter graphical interface.

This project was developed as an educational implementation of a 10-person face-recognition assignment. It is intended for learning classical computer-vision concepts—not as a production-grade biometric system.

## Features

- Desktop GUI built with Python and Tkinter
- SIFT keypoint and descriptor extraction using OpenCV
- Euclidean-distance matching with `cv2.BFMatcher(cv2.NORM_L2)`
- Lowe's ratio test to filter ambiguous descriptor matches
- Best-match identity prediction across enrolled training images
- Single-image recognition
- Batch evaluation on a labelled test folder
- Overall accuracy and per-person correct/total counts
- Beginner-friendly code with comments

## Assignment Dataset Format

The assignment specifies 10 selected people:

| Split | Images per person | Number of people | Total |
|---|---:|---:|---:|
| Training | 10 | 10 | 100 |
| Testing | 2 new images | 10 | 20 |

Use this folder structure:

```text
FaceRecognition_SIFT_Project/
├── app.py
├── requirements.txt
├── README.md
└── datasets/
    ├── train/
    │   ├── P1/
    │   ├── P2/
    │   ├── P3/
    │   ├── P4/
    │   ├── P5/
    │   ├── P6/
    │   ├── P7/
    │   ├── P8/
    │   ├── P9/
    │   └── P10/
    └── test/
        ├── P1/
        ├── P2/
        ├── P3/
        ├── P4/
        ├── P5/
        ├── P6/
        ├── P7/
        ├── P8/
        ├── P9/
        └── P10/
```

Each `train/Px` folder should contain 10 training images. Each `test/Px` folder should contain 2 new, independently captured images of the same person.

> **Important:** Do not use copies, crops, rotations, brightness-adjusted versions, or AI-generated derivatives of training images in the official test set. The assignment requires two new images per person. Reusing derivatives creates data leakage and makes the reported accuracy invalid as an independent test result.

## Requirements

- Python 3.10 or newer
- Visual Studio Code
- OpenCV with SIFT support
- NumPy
- Tkinter

Install the packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If required, install the OpenCV version used during development:

```bash
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install opencv-contrib-python==4.10.0.84
```

Do not install multiple OpenCV wheel variants into the same environment.

## Run the Application in Visual Studio Code

1. Download or clone this repository.
2. Open the project folder in Visual Studio Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

5. Install requirements:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Add training and independent test photographs to their respective folders.
7. Run the application:

```powershell
python app.py
```

8. In the application:

   - Select the training folder.
   - Click **Load Training Features**.
   - Click **Recognize an Image** to test an individual photograph.
   - Click **Evaluate Test Set** to evaluate all test photographs.

## How the Method Works

### 1. Face Region Preparation

The application attempts to detect frontal and profile faces using OpenCV Haar cascade classifiers.

It crops the detected face, resizes it, and converts it to grayscale before extracting SIFT features.

If the cascade XML files are unavailable or a face is not detected, the implementation may use a central-region crop fallback. The activity log reports this condition.

### 2. SIFT Feature Extraction

SIFT (Scale-Invariant Feature Transform) detects distinctive local image keypoints and computes descriptors for each keypoint.

OpenCV SIFT descriptors are 128-dimensional floating-point vectors.

### 3. Euclidean Distance Matching

The application uses OpenCV's Brute Force Matcher with the L2 norm:

```python
cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
```

Euclidean distance between descriptors A and B:

\[
d(A,B)=\sqrt{\sum_{i=1}^{128}(A_i-B_i)^2}
\]

The Lowe ratio test retains a match when:

\[
d_1 < 0.75 \times d_2
\]

Where:

- d1 = Distance to the nearest descriptor.
- d2 = Distance to the second-nearest descriptor.

### 4. Identity Prediction

The query photograph is matched against each stored training image.

The implementation:

1. Extracts SIFT descriptors from the query image.
2. Compares descriptors with the training images.
3. Counts good matches using Lowe's ratio test.
4. Keeps the three strongest image-level scores for each person.
5. Adds those scores to obtain an aggregate identity score.
6. Selects the person with the highest score, subject to a minimum-score rejection threshold.

This is a **SIFT descriptor-matching baseline**, not a neural-network classifier.

The training images are used as a reference gallery. The application does not learn model weights through gradient descent.

## Model Evaluation

The official assignment test set contains 20 independent test photographs: two per person.

For every test image:

1. The folder name (P1 to P10) is treated as the actual identity.
2. The application predicts an identity.
3. The predicted identity is compared with the actual identity.
4. Correct predictions are counted.

### Overall Accuracy

\[
Accuracy =
\frac{Correct\ Predictions}{Total\ Test\ Images}
\times 100
\]

For a 20-image test set:

\[
Accuracy =
\frac{Correct\ Predictions}{20}
\times 100
\]

Example:

If 16 out of 20 images are correctly identified:

\[
Accuracy = \frac{16}{20}\times100 = 80\%
\]

This is only an illustrative example. Report the actual result from your independent test run.

### Recognition Results Table

Fill this table using the actual evaluation log:

| Person | Test Images | Correct Predictions | Accuracy |
|---|---:|---:|---:|
| P1 | 2 | — | — |
| P2 | 2 | — | — |
| P3 | 2 | — | — |
| P4 | 2 | — | — |
| P5 | 2 | — | — |
| P6 | 2 | — | — |
| P7 | 2 | — | — |
| P8 | 2 | — | — |
| P9 | 2 | — | — |
| P10 | 2 | — | — |
| **Overall** | **20** | **—** | **—** |

Per-person accuracy:

\[
Person\ Accuracy =
\frac{Correct\ Predictions\ for\ Person}{2}
\times100
\]

The current application reports overall accuracy and per-person correct/total counts.

It does not currently calculate a full confusion matrix, precision, recall, or F1-score.

## Limitations

- SIFT is a general local-feature detector and descriptor, not a face-specific identity embedding.
- Changes in head pose, facial expression, lighting, image resolution, occlusion, and camera distance can reduce matching performance.
- Haar cascades may fail to detect profile, tilted, or partially occluded faces.
- Central-crop fallback is less reliable than a correctly detected face crop.
- Clothing, hair, accessories, and background may influence SIFT matching.
- Results depend on the quality and independence of training and testing photographs.
- This is an educational prototype and should not be used for security, access control, surveillance, or high-stakes biometric decisions.

## Suggested Screenshots

Include the following screenshots in your project report:

1. Dataset folder structure showing P1–P10.
2. Application home screen.
3. Training feature extraction log.
4. Single-image recognition result.
5. Test-set evaluation log.
6. Final recognition accuracy table.

## Learning Outcomes

By exploring this project, beginners can learn:

- How to organize a labelled image dataset.
- How local image features and SIFT descriptors work.
- How Euclidean distance compares image descriptors.
- How feature matching can be used for basic identification.
- How to build a desktop GUI using Tkinter.
- How to evaluate an identification pipeline using independent test images.

## Technologies Used

- Python
- OpenCV
- SIFT
- NumPy
- Tkinter
- Euclidean Distance
- Haar Cascade Face Detection

## Project Type

**Beginner-Friendly Computer Vision Project**

Built to make classical SIFT-based face recognition easier for beginners to understand and experiment with.

---

**Educational project:** Exploring classical computer vision through SIFT-based face recognition.
