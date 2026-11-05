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

## Assignment dataset format

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

Each `train/Px` folder should contain 10 training images. Each `test/Px` folder should contain 2 **new, independently captured** images of the same person.

> **Important evaluation rule:** Do not place copies, crops, rotations, brightness-adjusted versions, or AI-generated derivatives of training images in the official test set. The assignment asks for two new images per person. Reusing derivatives of training images creates data leakage and makes the reported accuracy invalid as an independent test result. If you use a synthetic/augmented test set to check that the program runs, label those results as a software demonstration only—not final assignment accuracy.

## Requirements

- Python 3.10 or newer
- Visual Studio Code (or another Python IDE)
- OpenCV with SIFT support
- NumPy
- Tkinter (usually included with standard Windows Python installations)

Install the packages from the project folder:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need to install the OpenCV build used during development:

```bash
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install opencv-contrib-python==4.10.0.84
```

Do not install multiple OpenCV wheel variants into the same environment at the same time.

## Run the application in Visual Studio Code

1. Download or clone this repository.
2. Open the project folder in Visual Studio Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment (Windows PowerShell):

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

5. Install the requirements:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Put the training and independent test images into the folders shown above.
7. Start the GUI:

   ```powershell
   python app.py
   ```

8. In the application:
   - Select the training folder if it is not already selected.
   - Click **Load Training Features**.
   - Click **Recognize an Image** to try one image.
   - Click **Evaluate Test Set** to evaluate all images inside the test folders.

## How the method works

### 1. Face region preparation

The application attempts to detect frontal/profile faces with OpenCV Haar cascade classifiers. It crops the detected face, resizes it, and converts it to grayscale before extracting SIFT features.

If the cascade XML files are unavailable or a face is not detected, the current implementation may use a central-region crop fallback. The log reports this condition. Such fallback cases should be inspected because the crop may include background or clothing features.

### 2. SIFT feature extraction

SIFT detects distinctive local image keypoints and computes a descriptor for each keypoint. OpenCV SIFT descriptors are 128-dimensional floating-point vectors.

### 3. Euclidean-distance matching

The application uses OpenCV's brute-force matcher with the L2 norm:

```python
cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
```

For a pair of descriptors \(A\) and \(B\), Euclidean distance is:

\[
d(A,B)=\sqrt{\sum_{i=1}^{128}(A_i-B_i)^2}
\]

The Lowe ratio test retains a match when:

\[
d_1 < 0.75\,d_2
\]

where \(d_1\) and \(d_2\) are the distances to the nearest and second-nearest candidate descriptors.

### 4. Identity prediction

The query image is matched against each stored training image. The implementation counts good matches, keeps the three strongest image-level scores for each person, and sums those three scores. The person with the highest aggregate score is selected, subject to a small minimum-score rejection threshold.

This is a **SIFT descriptor-matching baseline**, not a neural-network classifier. The training images are used as an enrolled reference gallery; the application does not learn model weights through gradient descent.

## Model evaluation

For the official assignment test set, there should be 20 independent test images: two per person.

For every test image:

1. The folder name (`P1` ... `P10`) is treated as the ground-truth label.
2. The application predicts an identity.
3. The prediction is compared with the ground-truth label.
4. A correct prediction is counted when the predicted label equals the folder label.

### Overall accuracy

\[
\text{Accuracy}=\frac{\text{Number of correct predictions}}{\text{Total test images}}\times100
\]

For a 20-image test set:

\[
\text{Accuracy}=\frac{\text{Correct predictions}}{20}\times100
\]

Example only: if 16 of 20 independent test images are correctly identified:

\[
\text{Accuracy}=\frac{16}{20}\times100=80\%
\]

Do not use this example as your actual result. Report the number produced by your own independent test run.

### Result table template

Fill this table using the actual application evaluation log:

| Person | Test images | Correct predictions | Accuracy |
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
\text{Person accuracy}=\frac{\text{Correct predictions for that person}}{2}\times100
\]

The current GUI reports overall accuracy and per-person correct/total counts. It does not currently calculate a full confusion matrix, precision, recall, or F1-score. Those should not be claimed unless you add and run those calculations.

## Limitations

- SIFT is a general local-feature detector/descriptor, not a face-specific identity embedding.
- Changes in head pose, facial expression, lighting, image resolution, occlusion, and camera distance can reduce matching performance.
- Haar cascades may fail to detect profile, tilted, or partially occluded faces.
- The central-crop fallback is less reliable than a correctly detected face crop.
- Clothing, hair, accessories, and background may still influence SIFT matches, especially when face detection fails or the crop contains non-face regions.
- Results depend on the quality and independence of the collected training and test photographs.
- This is an educational prototype and should not be used for security, access control, surveillance, or other high-stakes biometric decisions.

## Suggested screenshots for the project report

1. Dataset folder structure showing P1–P10.
2. Application home screen.
3. Training feature extraction log.
4. Single-image recognition result.
5. Test-set evaluation log.
6. Final result table calculated from the independent 20-image test set.

## Repository topics

`computer-vision` `face-recognition` `sift` `opencv` `python` `tkinter` `beginner-friendly` `image-processing` `euclidean-distance`

## Learning outcomes

By exploring this project, beginners can learn:

- How to organize a labelled image dataset
- How local image features and SIFT descriptors work
- How Euclidean distance is used to compare descriptors
- How feature matching can be used for basic identification
- How to build a desktop GUI with Tkinter
- How to evaluate a classifier-like identification pipeline using held-out test images

---

**Educational project:** Built to make classical SIFT-based face recognition easier for beginners to understand and experiment with.
