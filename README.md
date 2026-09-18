# VisionMetrix 3D (VM3D)

## Computer Vision Based 3D Inspection and Analysis System

**Author:** Mitali Gautam

VisionMetrix 3D (VM3D) is a Python-based Computer Vision project developed to bring together the major concepts covered in the Computer Vision course into one application.

The project focuses on image processing, feature extraction, stereo vision, 3D reconstruction, image segmentation, pattern analysis and motion analysis. It uses synthetic inspection data so that different computer vision techniques can be tested in a controlled and repeatable way.

The main aim of the project is to understand how individual computer vision algorithms work and how they can be combined to form a complete image analysis and inspection pipeline.

---

## Project Overview

The project is organised into four major modules based on the Computer Vision syllabus:

1. **Low-Level Image Processing**  
   Includes image transformations, filtering, Fourier transform, enhancement and restoration techniques.

2. **Stereo Vision and 3D Processing**  
   Includes camera calibration, epipolar geometry, disparity estimation, 3D reconstruction and photometric stereo.

3. **Feature Extraction and Image Segmentation**  
   Includes edge detection, line detection, corner detection, feature descriptors, scale-space analysis and segmentation techniques.

4. **Pattern and Motion Analysis**  
   Includes clustering, classification, dimensionality reduction and motion analysis techniques such as optical flow.

The project also provides a command-line interface, a simple graphical dashboard and SQLite-based storage for inspection-related information.

---

## Features

### Image Processing

- Geometric image transformations
- Image filtering
- Fourier-domain processing
- Image enhancement
- Image restoration

### Stereo Vision and 3D Processing

- Camera calibration
- Epipolar geometry
- Stereo disparity estimation
- 3D reconstruction
- Photometric stereo

### Feature Extraction

- Canny edge detection
- LoG (Laplacian of Gaussian)
- DoG (Difference of Gaussian)
- Hough Transform
- Harris corner detection
- Feature descriptors
- Scale-space analysis
- Image pyramids
- Gabor filters
- DWT

### Image Segmentation

- Region-based segmentation
- Edge-based segmentation
- Thresholding and related segmentation techniques

### Pattern Analysis

- K-Means clustering
- Gaussian Mixture Model
- KNN classification
- Naive Bayes classification
- PCA
- LDA

### Motion Analysis

- Background subtraction
- Optical flow
- Motion tracking

### Other Features

- Synthetic data generation for testing
- Command-line interface
- Graphical dashboard
- SQLite database storage
- Automated testing using Pytest
- Generation and storage of processed outputs

---

## Technologies and Tools Used

| Technology / Tool | Purpose |
|---|---|
| Python 3.10+ | Main programming language |
| OpenCV | Image processing and computer vision |
| NumPy | Numerical and array operations |
| SciPy | Scientific and mathematical operations |
| Scikit-image | Image processing algorithms |
| Scikit-learn | Clustering, classification and dimensionality reduction |
| Matplotlib | Data and image visualization |
| Tkinter | Graphical dashboard |
| SQLite | Local database storage |
| Pytest | Automated testing |
| ReportLab | Report generation |

---

## Project Structure

```text
VisionMetrix3D/
│
├── data/
│   └── Input and synthetic data
│
├── docs/
│   └── Project documentation and diagrams
│
├── output/
│   └── Generated results and visual outputs
│
├── tests/
│   └── Automated test files
│
├── visionmetrix/
│   ├── lowlevel/
│   ├── stereo3d/
│   ├── features/
│   ├── pattern_motion/
│   ├── storage/
│   ├── pipeline/
│   └── data/
│
├── cli.py
├── dashboard.py
├── requirements.txt
├── statement.md
├── README.md
└── .gitignore
## Installation

### Requirements

Before running the project, make sure you have:

- Python 3.10 or higher
- pip
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mitali66/VisionMetrix3D_24BAI10762.git
cd VisionMetrix3D_24BAI10762
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\activate
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Run the Complete Demonstration

The main project demonstration can be run using:

```bash
python cli.py demo
```

This runs the main computer vision pipeline using synthetic inspection data and generates the corresponding results in the `output` directory.

### 2. Run Photometric Stereo

```bash
python cli.py photometric --size 256
```

This runs the photometric stereo demonstration.

### 3. Launch the Dashboard

```bash
python dashboard.py
```

This opens the graphical dashboard of the project.

### 4. Inspect a Stereo Image Pair

A custom stereo image pair can be processed using:

```bash
python cli.py inspect --image path/to/left.png --right path/to/right.png --item-id PCB_101
```

Replace the image paths and item ID with the required input values.

---

## Testing

Automated tests are included in the `tests` directory.

To run all tests:

```bash
python -m pytest -v
```

The command displays the individual test results along with the final number of passed and failed tests.

Testing is used to verify the functionality of different modules and parts of the computer vision pipeline.

---

## Screenshots

The project includes visual outputs that can be used to demonstrate the working of different computer vision techniques.

Examples of results include:

- Image preprocessing results
- Edge detection results
- Feature detection
- Image segmentation
- Stereo and disparity results
- 3D reconstruction and point clouds
- Photometric stereo results
- Optical flow and motion analysis
- Dashboard interface

Generated result images are stored in the `output` directory.

---

## Output

Depending on the command being executed, the project can generate:

- Processed images
- Edge detection results
- Feature detection results
- Segmentation results
- Stereo images
- Disparity maps
- Depth-related results
- 3D point-cloud files
- Photometric stereo results
- Optical flow results
- Motion analysis results
- Inspection visualizations

These outputs help in understanding how different computer vision techniques process the input data.

---

## Project Documentation

The `docs` directory contains the documentation and design diagrams related to the project.

It includes:

- System Architecture
- Workflow Diagram
- Use Case Diagram
- ER Diagram
- Supporting documentation

These diagrams describe the structure and working flow of VisionMetrix 3D.

---

## System Workflow

```text
Input / Synthetic Data
        ↓
Image Preprocessing
        ↓
Feature Extraction
        ↓
Stereo / 3D Processing
        ↓
Pattern & Motion Analysis
        ↓
Inspection Results
        ↓
Storage and Visualization
```

---

## Database

The project uses **SQLite** for local storage of inspection-related information.

SQLite allows the project to store relevant inspection data without requiring a separate database server.

---

## Testing Approach

The project uses **Pytest** for automated testing.

The testing process includes:

- Testing individual modules
- Checking implemented computer vision algorithms
- Testing different parts of the processing pipeline
- Verifying expected outputs
- Checking the integration between project components

Tests can be executed using:

```bash
python -m pytest -v
```

---

## Limitations

The current version of the project has some limitations:

- The main demonstrations use synthetic inspection data.
- Real camera and hardware integration is not the primary focus.
- Processing time depends on image size and system hardware.
- 3D reconstruction results depend on the quality of stereo input and calibration.
- The dashboard is designed mainly as an academic demonstration interface.

---

## Future Enhancements

Possible future improvements include:

- Integration with real stereo cameras
- Real-time image processing
- Improved depth estimation
- Machine learning based defect detection
- Improved 3D point-cloud visualization
- More interactive dashboard features
- Support for larger real-world datasets
- Cloud-based storage and monitoring
- Improved processing performance

---

## Academic Scope

VisionMetrix 3D was developed as an academic Computer Vision project to demonstrate the concepts covered in the course.

The project combines multiple computer vision techniques into a single application rather than treating every technique as a completely separate program.

Synthetic data is used where required to provide consistent and repeatable testing.

The project is intended for learning, experimentation and demonstration of computer vision concepts.

---

## Author

**Mitali Gautam**
---

## References

- Python Documentation
- OpenCV Documentation
- NumPy Documentation
- SciPy Documentation
- Scikit-image Documentation
- Scikit-learn Documentation
- Matplotlib Documentation
- Pytest Documentation
- Course lecture material and reference notes

---
