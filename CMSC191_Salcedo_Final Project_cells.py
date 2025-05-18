# CMSC191 Final Project: Face and Feature Shape Analyzer
#
# Dependencies:
# - opencv-python
# - pyqt5
# - numpy
# - dlib (for facial landmark detection)
#
# Before running, install dependencies:
# pip install opencv-python pyqt5 numpy dlib

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QCheckBox, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import dlib

# Load dlib's pre-trained face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'shape_predictor_68_face_landmarks.dat')
landmark_predictor = dlib.shape_predictor(model_path)

# --- KNN Digit Recognizer Setup ---
digits_img = cv2.imread(os.path.join(script_dir, 'cnn', 'digits.png'))
gray_digits = cv2.cvtColor(digits_img, cv2.COLOR_BGR2GRAY)
cells = [np.hsplit(row,100) for row in np.vsplit(gray_digits,50)]
x = np.array(cells)
train = x[:,:70].reshape(-1,400).astype(np.float32)
train_labels = np.repeat(np.arange(10),350)[:,np.newaxis]
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
# --- End KNN Setup ---

# Helper functions for digit ROI extraction

def x_cord_contour(contour):
    # Returns the x coordinate of the bounding rectangle of a contour
    M = cv2.moments(contour)
    if M['m00'] != 0:
        return int(M['m10']/M['m00'])
    else:
        return 0

def makeSquare(not_square):
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        if (height > width):
            pad = int((height - width)/2)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad = int((width - height)/2)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=BLACK)
        return doublesize_square

def resize_to_pixel(dimensions, image):
    buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    return ReSizedImg

# Helper function to convert dlib shape to numpy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def analyze_face_shape(landmarks):
    # jaw width (points 4 to 12)
    jaw_width = np.linalg.norm(landmarks[4] - landmarks[12])
    # cheekbone width (points 1 to 15)
    cheekbone_width = np.linalg.norm(landmarks[1] - landmarks[15])
    # forehead width (approximate: points 17 to 26)
    forehead_width = np.linalg.norm(landmarks[17] - landmarks[26])
    # face height (chin to midpoint between eyebrows)
    chin = landmarks[8]
    brow_mid = ((landmarks[21] + landmarks[22]) // 2)
    face_height = np.linalg.norm(chin - brow_mid)
    # ratios
    jaw_to_height = jaw_width / face_height
    cheek_to_height = cheekbone_width / face_height
    forehead_to_height = forehead_width / face_height
    # heuristic rules for face shape
    if abs(jaw_to_height - cheek_to_height) < 0.05 and abs(cheek_to_height - forehead_to_height) < 0.05:
        return "Square"
    elif cheek_to_height > jaw_to_height and cheek_to_height > forehead_to_height:
        return "Diamond"
    elif jaw_to_height > cheek_to_height and jaw_to_height > forehead_to_height:
        return "Triangle"
    elif forehead_to_height > cheek_to_height and forehead_to_height > jaw_to_height:
        return "Heart"
    elif face_height / cheekbone_width > 1.5:
        return "Oval"
    else:
        return "Round"

def analyze_lip_shape(landmarks):
    lips = landmarks[48:68]
    width = np.linalg.norm(lips[0] - lips[6])
    upper_height = np.linalg.norm(lips[3] - lips[13])
    lower_height = np.linalg.norm(lips[9] - lips[19])
    avg_height = (upper_height + lower_height) / 2
    ratio = width / avg_height if avg_height != 0 else 0
    if ratio > 2.2:
        return "Wide"
    elif ratio < 1.4:
        return "Full"
    else:
        return "Medium"

def analyze_nose_shape(landmarks):
    nose = landmarks[27:36]
    width = np.linalg.norm(nose[0] - nose[4])
    height = np.linalg.norm(nose[0] - nose[6])
    nostril_flare = np.linalg.norm(landmarks[31] - landmarks[35])
    ratio = width / height if height != 0 else 0
    if nostril_flare / width > 0.5:
        return "Flared"
    elif ratio > 0.9:
        return "Wide"
    elif ratio < 0.6:
        return "Narrow"
    else:
        return "Medium"

def analyze_eye_shape(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    def eye_features(eye):
        width = np.linalg.norm(eye[0] - eye[3])
        height = (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / 2
        area = 0.5 * abs(
            sum(eye[i][0]*eye[(i+1)%6][1] - eye[(i+1)%6][0]*eye[i][1] for i in range(6))
        )
        ratio = width / height if height != 0 else 0
        return ratio, area
    left_ratio, left_area = eye_features(left_eye)
    right_ratio, right_area = eye_features(right_eye)
    avg_ratio = (left_ratio + right_ratio) / 2
    avg_area = (left_area + right_area) / 2
    if avg_ratio > 2.5:
        return "Almond"
    elif avg_area > 400:
        return "Large"
    elif avg_area < 200:
        return "Small"
    else:
        return "Round"

class MainMenu(QWidget):
    def __init__(self, ocr_callback, face_callback):
        super().__init__()
        self.setWindowTitle('Main Menu')
        layout = QVBoxLayout()
        self.ocr_btn = QPushButton('OCR Numbers')
        self.face_btn = QPushButton('Analyze Face Features')
        self.exit_btn = QPushButton('Exit')
        layout.addWidget(self.ocr_btn)
        layout.addWidget(self.face_btn)
        layout.addWidget(self.exit_btn)
        self.setLayout(layout)
        self.ocr_btn.clicked.connect(ocr_callback)
        self.face_btn.clicked.connect(face_callback)
        self.exit_btn.clicked.connect(self.exit_all)
        self.resize(400, 300)

    def exit_all(self):
        QApplication.quit()

class OCRWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle('OCR Numbers')
        self.image_label = QLabel()
        self.result_label = QLabel('Detected Number:')
        self.back_btn = QPushButton('Back to Main Menu')
        self.back_btn.clicked.connect(back_callback)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = x_cord_contour, reverse = False)
        full_number = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w >= 5 and h >= 25:
                roi = blurred[y:y + h, x:x + w]
                ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
                squared = makeSquare(roi)
                final = resize_to_pixel(20, squared)
                final_array = final.reshape((1,400)).astype(np.float32)
                ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
                number = str(int(result[0, 0]))
                full_number.append(number)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, number, (x , y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        self.result_label.setText('Detected Number: ' + ''.join(full_number))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        cv2.destroyAllWindows()
        event.accept()

class FaceAnalyzerApp(QWidget):
    def __init__(self, back_callback=None):
        super().__init__()
        self.setWindowTitle('Face & Feature Shape Analyzer')
        self.image_label = QLabel()
        self.result_label = QLabel('Results will appear here.')
        self.ocr_label = QLabel('OCR:')
        self.cb_face_shape = QCheckBox('Show Face Shape Label')
        self.cb_face_shape.setChecked(True)
        self.cb_lip_shape = QCheckBox('Show Lip Shape Label')
        self.cb_lip_shape.setChecked(False)
        self.cb_nose_shape = QCheckBox('Show Nose Shape Label')
        self.cb_nose_shape.setChecked(False)
        self.cb_eye_shape = QCheckBox('Show Eye Shape Label')
        self.cb_eye_shape.setChecked(False)
        self.cb_landmarks = QCheckBox('Show Landmark Points')
        self.cb_landmarks.setChecked(False)
        toggles_layout = QHBoxLayout()
        toggles_layout.addWidget(self.cb_face_shape)
        toggles_layout.addWidget(self.cb_lip_shape)
        toggles_layout.addWidget(self.cb_nose_shape)
        toggles_layout.addWidget(self.cb_eye_shape)
        toggles_layout.addWidget(self.cb_landmarks)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.ocr_label)
        layout.addLayout(toggles_layout)
        if back_callback:
            self.back_btn = QPushButton('Back to Main Menu')
            self.back_btn.clicked.connect(back_callback)
            layout.addWidget(self.back_btn)
        self.setLayout(layout)
        self.resize(900, 700)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cb_face_shape.stateChanged.connect(self.update_frame)
        self.cb_lip_shape.stateChanged.connect(self.update_frame)
        self.cb_nose_shape.stateChanged.connect(self.update_frame)
        self.cb_eye_shape.stateChanged.connect(self.update_frame)
        self.cb_landmarks.stateChanged.connect(self.update_frame)
        self.start_webcam()

    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.timer.isActive():
            self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        orig_frame = frame.copy()
        gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        result_text = "No face detected."
        ocr_text = ""
        for face in faces:
            shape = landmark_predictor(gray, face)
            landmarks = shape_to_np(shape)
            # Draw overlays for features if their label is checked
            if self.cb_face_shape.isChecked():
                # Outline jaw/face
                face_outline = np.array(landmarks[0:17], np.int32)
                cv2.polylines(frame, [face_outline], False, (0,0,255), 2)
                cv2.putText(frame, f"Face: {analyze_face_shape(landmarks)}", (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)      
            if self.cb_lip_shape.isChecked():
                lips_outline = np.array(landmarks[48:60], np.int32)
                cv2.polylines(frame, [lips_outline], True, (255,0,255), 2)
                lip_center = tuple(np.mean(landmarks[48:68], axis=0).astype(int))
                cv2.putText(frame, f"Lips: {analyze_lip_shape(landmarks)}", (lip_center[0]-40, lip_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            if self.cb_nose_shape.isChecked():
                nose_outline = np.array(landmarks[27:36], np.int32)
                cv2.polylines(frame, [nose_outline], False, (0,128,255), 2)
                nose_center = tuple(np.mean(landmarks[27:36], axis=0).astype(int))
                cv2.putText(frame, f"Nose: {analyze_nose_shape(landmarks)}", (nose_center[0]-30, nose_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)
            if self.cb_eye_shape.isChecked():
                left_eye_outline = np.array(landmarks[36:42], np.int32)
                right_eye_outline = np.array(landmarks[42:48], np.int32)
                cv2.polylines(frame, [left_eye_outline], True, (0,255,255), 2)
                cv2.polylines(frame, [right_eye_outline], True, (0,255,255), 2)
                left_eye_center = tuple(np.mean(landmarks[36:42], axis=0).astype(int))
                right_eye_center = tuple(np.mean(landmarks[42:48], axis=0).astype(int))
                cv2.putText(frame, f"Eye: {analyze_eye_shape(landmarks)}", (left_eye_center[0]-20, left_eye_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.putText(frame, f"Eye: {analyze_eye_shape(landmarks)}", (right_eye_center[0]-20, right_eye_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            if self.cb_landmarks.isChecked():
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            result_text = f"Face: {analyze_face_shape(landmarks)}\nLips: {analyze_lip_shape(landmarks)}\nNose: {analyze_nose_shape(landmarks)}\nEyes: {analyze_eye_shape(landmarks)}"
            break
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        self.result_label.setText(result_text)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        cv2.destroyAllWindows()
        event.accept()

class MainController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.menu = MainMenu(self.show_ocr, self.show_face)
        self.ocr_window = None
        self.face_window = None
        self.menu.show()

    def show_ocr(self):
        if self.ocr_window is None:
            self.ocr_window = OCRWindow(self.show_menu)
        self.menu.hide()
        self.ocr_window.show()

    def show_face(self):
        if self.face_window is None:
            self.face_window = FaceAnalyzerApp(self.show_menu)
        self.menu.hide()
        self.face_window.show()

    def show_menu(self):
        if self.ocr_window:
            self.ocr_window.hide()
        if self.face_window:
            self.face_window.hide()
        self.menu.show()

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == '__main__':
    try:
        controller = MainController()
        controller.run()
    except SystemExit:
        # Release OpenCV resources and suppress traceback in interactive environments
        try:
            # Make sure all webcam resources are released
            if hasattr(controller, 'ocr_window') and controller.ocr_window and controller.ocr_window.cap:
                controller.ocr_window.cap.release()
            if hasattr(controller, 'face_window') and controller.face_window and controller.face_window.cap:
                controller.face_window.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass