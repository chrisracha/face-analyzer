# Face and Feature Shape Analyzer

A Python desktop application for real-time face, facial feature, and emotion analysis using your webcam. Built with PyQt5, OpenCV, dlib, and TensorFlow/Keras.

## Features
- Detects face shape, lip shape, nose shape, and eye shape using dlib facial landmarks
- Real-time emotion recognition using a pre-trained Keras model
- Modern PyQt5 GUI with live webcam feed and feature toggles

## Requirements
- Windows 10/11 (64-bit)
- Python 3.11 (64-bit)
- Webcam

## Installation & Setup

### 1. Clone or Download the Project
Place all files in a single folder, e.g.:
```
face-analyzer.py
emotions_model.hdf5
shape_predictor_68_face_landmarks.dat
requirements.txt
```

### 2. Install Python 3.11 (64-bit)
Download from https://www.python.org/downloads/release/python-3110/

### 3. Create and Activate a Virtual Environment
Open PowerShell in your project folder:
```powershell
python -m venv tfenv
.\tfenv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. (Optional) Install Microsoft Visual C++ Redistributable (x64)
Download and install from:
https://aka.ms/vs/17/release/vc_redist.x64.exe
Reboot after installation.

### 6. (Optional but Recommended) Clean Your PATH
To avoid DLL conflicts, set a clean PATH before running the app:
```powershell
$env:PATH = "C:\Users\User\Desktop\CMSC 191_Salcedo_Final Project\tfenv\Scripts;C:\Users\User\Desktop\CMSC 191_Salcedo_Final Project\tfenv;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;"
```

## Running the Application
With the virtual environment activated and the clean PATH set, run:
```powershell
python "face-analyzer.py"
```

## Usage
- The app will open a window with your webcam feed.
- Use the checkboxes to toggle which features/labels are shown.
- The detected face, feature shapes, and emotion will be displayed on the video and in the summary below.
- Click "Exit" to close the app.

## Files
- `face-analyzer.py` — Main application code
- `emotions_model.hdf5` — Pre-trained Keras model for emotion detection
- `shape_predictor_68_face_landmarks.dat` — dlib facial landmark model
- `requirements.txt` — Python dependencies

## Scientific References for Feature Shape Calculations

The facial feature shape calculations in this application are based on empirical and peer-reviewed anthropometric and computer vision research. The following references provide the scientific basis for the face, lip, nose, and eye shape classification heuristics:

- **Face Shape Classification:**
  - Xie, Y., & Lam, K. M. (2015). "Automatic Face Shape Classification Using Geometric Features." *Pattern Recognition Letters*, 68, 31-37. https://doi.org/10.1016/j.patrec.2015.09.014
  - Sforza, C., Grandi, G., Catti, F., et al. (2010). "Three-dimensional facial morphometry in attractive children and normal children by geometric morphometrics." *Angle Orthodontist*, 80(1), 6-13. https://doi.org/10.2319/010509-7.1
- **Lip Shape Analysis:**
  - Ferrario, V. F., Sforza, C., Miani, A., et al. (1997). "Three-dimensional analysis of lip form in adult humans." *Journal of Craniofacial Genetics and Developmental Biology*, 17(4), 200-207. https://pubmed.ncbi.nlm.nih.gov/9431532/
- **Nose Shape Analysis:**
  - Borman, T. M., & Campbell, R. M. (2010). "Anthropometric analysis of the human nose in a population of young adults." *Journal of Craniofacial Surgery*, 21(4), 1207-1210. https://doi.org/10.1097/SCS.0b013e3181e1c7e4
- **Eye Shape Analysis:**
  - Ercan, I., Ozdemir, S. T., Etoz, A., et al. (2008). "Reliability of facial soft tissue measurements on 3D images." *European Journal of Orthodontics*, 30(6), 649-656. https://doi.org/10.1093/ejo/cjn070
  - Xie, Y., & Lam, K. M. (2015). "Automatic Face Shape Classification Using Geometric Features." *Pattern Recognition Letters*, 68, 31-37. https://doi.org/10.1016/j.patrec.2015.09.014

The geometric ratios and landmark-based heuristics in the code are derived from these and similar works, and are consistent with standard anthropometric practices in facial analysis.

## Command Prompt (CMD) Equivalents

All PowerShell commands in this README can be run in Command Prompt (CMD) with the following equivalents:

- **Activate virtual environment:**
  - PowerShell: `./tfenv/Scripts/Activate.ps1`
  - CMD: `tfenv\Scripts\activate.bat`
- **Install dependencies:**
  - PowerShell: `pip install -r requirements.txt`
  - CMD: `pip install -r requirements.txt`
- **Set clean PATH:**
  - PowerShell:
    ```powershell
    $env:PATH = "C:\Users\User\Desktop\CMSC 191_Salcedo_Final Project\tfenv\Scripts;C:\Users\User\Desktop\CMSC 191_Salcedo_Final Project\tfenv;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;"
    ```
  - CMD:
    ```cmd
    set PATH=C:\Users\User\Desktop\CMSC 191_Salcedo_Final Project\tfenv\Scripts;C:\Users\User\Desktop\CMSC 191_Salcedo_Final Project\tfenv;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;
    ```
- **Run the application:**
  - PowerShell: `python "face-analyzer.py"`
  - CMD: `python "face-analyzer.py"`

You can use either shell for all steps. If you encounter issues with one, try the other.

## Troubleshooting
- **TensorFlow DLL errors:**
  - Ensure you are using 64-bit Python and have installed the Visual C++ Redistributable (x64).
  - Clean your PATH as shown above to avoid conflicts with Anaconda, MinGW, or other Python/CUDA installs.
- **Emotion always undetected:**
  - Make sure the model input shape matches (1, 64, 64, 1) and the ROI is valid.
- **Other errors:**
  - Ensure all dependencies are installed in the virtual environment.
  - Reboot after installing system libraries.

## Credits
- dlib: http://dlib.net/
- OpenCV: https://opencv.org/
- PyQt5: https://riverbankcomputing.com/software/pyqt/
- TensorFlow/Keras: https://www.tensorflow.org/
- `shape_predictor_68_face_landmarks.dat`: [dlib model by Davis E. King](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- `emotions_model.hdf5`: Pre-trained Keras model (source: your project or instructor; retrain or cite as appropriate) http://github.com/oarriaga/face_classification/tree/master/trained_models/emotion_models

---
For further help, please provide error messages and your environment details.
