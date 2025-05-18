[33mcommit 913895b0127668c16c46500c78a11a3a66eee1a8[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m)[m
Author: chris samuel <cosalcedo@up.edu.ph>
Date:   Mon May 19 05:44:17 2025 +0800

    Fix face analyzer app issues and implement proper error handling

[1mdiff --git a/CMSC191_Salcedo_Final Project_cells.py b/CMSC191_Salcedo_Final Project_cells.py[m
[1mindex 9a6ecb5..e69de29 100644[m
[1m--- a/CMSC191_Salcedo_Final Project_cells.py[m	
[1m+++ b/CMSC191_Salcedo_Final Project_cells.py[m	
[36m@@ -1,407 +0,0 @@[m
[31m-# %% [markdown][m
[31m-# CMSC191 Final Project: Face and Feature Shape Analyzer[m
[31m-#[m
[31m-# Dependencies:[m
[31m-# - opencv-python[m
[31m-# - pyqt5[m
[31m-# - numpy[m
[31m-# - dlib (for facial landmark detection)[m
[31m-#[m
[31m-# Before running, install dependencies:[m
[31m-# pip install opencv-python pyqt5 numpy dlib[m
[31m-[m
[31m-# %%[m
[31m-import sys[m
[31m-import os[m
[31m-import cv2[m
[31m-import numpy as np[m
[31m-from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QCheckBox, QHBoxLayout[m
[31m-from PyQt5.QtGui import QImage, QPixmap[m
[31m-from PyQt5.QtCore import QTimer[m
[31m-import dlib[m
[31m-[m
[31m-# %%[m
[31m-# Load dlib's pre-trained face detector and shape predictor[m
[31m-face_detector = dlib.get_frontal_face_detector()[m
[31m-script_dir = os.path.dirname(os.path.abspath(__file__))[m
[31m-model_path = os.path.join(script_dir, 'shape_predictor_68_face_landmarks.dat')[m
[31m-landmark_predictor = dlib.shape_predictor(model_path)[m
[31m-[m
[31m-# --- KNN Digit Recognizer Setup ---[m
[31m-digits_img = cv2.imread(os.path.join(script_dir, 'cnn', 'digits.png'))[m
[31m-gray_digits = cv2.cvtColor(digits_img, cv2.COLOR_BGR2GRAY)[m
[31m-cells = [np.hsplit(row,100) for row in np.vsplit(gray_digits,50)][m
[31m-x = np.array(cells)[m
[31m-train = x[:,:70].reshape(-1,400).astype(np.float32)[m
[31m-train_labels = np.repeat(np.arange(10),350)[:,np.newaxis][m
[31m-knn = cv2.ml.KNearest_create()[m
[31m-knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)[m
[31m-# --- End KNN Setup ---[m
[31m-[m
[31m-# Helper functions for digit ROI extraction[m
[31m-[m
[31m-def x_cord_contour(contour):[m
[31m-    # Returns the x coordinate of the bounding rectangle of a contour[m
[31m-    M = cv2.moments(contour)[m
[31m-    if M['m00'] != 0:[m
[31m-        return int(M['m10']/M['m00'])[m
[31m-    else:[m
[31m-        return 0[m
[31m-[m
[31m-def makeSquare(not_square):[m
[31m-    BLACK = [0,0,0][m
[31m-    img_dim = not_square.shape[m
[31m-    height = img_dim[0][m
[31m-    width = img_dim[1][m
[31m-    if (height == width):[m
[31m-        square = not_square[m
[31m-        return square[m
[31m-    else:[m
[31m-        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)[m
[31m-        height = height * 2[m
[31m-        width = width * 2[m
[31m-        if (height > width):[m
[31m-            pad = int((height - width)/2)[m
[31m-            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)[m
[31m-        else:[m
[31m-            pad = int((width - height)/2)[m
[31m-            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=BLACK)[m
[31m-        return doublesize_square[m
[31m-[m
[31m-def resize_to_pixel(dimensions, image):[m
[31m-    buffer_pix = 4[m
[31m-    dimensions  = dimensions - buffer_pix[m
[31m-    squared = image[m
[31m-    r = float(dimensions) / squared.shape[1][m
[31m-    dim = (dimensions, int(squared.shape[0] * r))[m
[31m-    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)[m
[31m-    img_dim2 = resized.shape[m
[31m-    height_r = img_dim2[0][m
[31m-    width_r = img_dim2[1][m
[31m-    BLACK = [0,0,0][m
[31m-    if (height_r > width_r):[m
[31m-        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)[m
[31m-    if (height_r < width_r):[m
[31m-        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)[m
[31m-    p = 2[m
[31m-    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)[m
[31m-    return ReSizedImg[m
[31m-[m
[31m-# %%[m
[31m-# Helper function to convert dlib shape to numpy array[m
[31m-def shape_to_np(shape, dtype="int"):[m
[31m-    coords = np.zeros((68, 2), dtype=dtype)[m
[31m-    for i in range(0, 68):[m
[31m-        coords[i] = (shape.part(i).x, shape.part(i).y)[m
[31m-    return coords[m
[31m-[m
[31m-# %%[m
[31m-def analyze_face_shape(landmarks):[m
[31m-    # jaw width (points 4 to 12)[m
[31m-    jaw_width = np.linalg.norm(landmarks[4] - landmarks[12])[m
[31m-    # cheekbone width (points 1 to 15)[m
[31m-    cheekbone_width = np.linalg.norm(landmarks[1] - landmarks[15])[m
[31m-    # forehead width (approximate: points 17 to 26)[m
[31m-    forehead_width = np.linalg.norm(landmarks[17] - landmarks[26])[m
[31m-    # face height (chin to midpoint between eyebrows)[m
[31m-    chin = landmarks[8][m
[31m-    brow_mid = ((landmarks[21] + landmarks[22]) // 2)[m
[31m-    face_height = np.linalg.norm(chin - brow_mid)[m
[31m-    # ratios[m
[31m-    jaw_to_height = jaw_width / face_height[m
[31m-    cheek_to_height = cheekbone_width / face_height[m
[31m-    forehead_to_height = forehead_width / face_height[m
[31m-    # heuristic rules for face shape[m
[31m-    if abs(jaw_to_height - cheek_to_height) < 0.05 and abs(cheek_to_height - forehead_to_height) < 0.05:[m
[31m-        return "Square"[m
[31m-    elif cheek_to_height > jaw_to_height and cheek_to_height > forehead_to_height:[m
[31m-        return "Diamond"[m
[31m-    elif jaw_to_height > cheek_to_height and jaw_to_height > forehead_to_height:[m
[31m-        return "Triangle"[m
[31m-    elif forehead_to_height > cheek_to_height and forehead_to_height > jaw_to_height:[m
[31m-        return "Heart"[m
[31m-    elif face_height / cheekbone_width > 1.5:[m
[31m-        return "Oval"[m
[31m-    else:[m
[31m-        return "Round"[m
[31m-[m
[31m-def analyze_lip_shape(landmarks):[m
[31m-    lips = landmarks[48:68][m
[31m-    width = np.linalg.norm(lips[0] - lips[6])[m
[31m-    upper_height = np.linalg.norm(lips[3] - lips[13])[m
[31m-    lower_height = np.linalg.norm(lips[9] - lips[19])[m
[31m-    avg_height = (upper_height + lower_height) / 2[m
[31m-    ratio = width / avg_height if avg_height != 0 else 0[m
[31m-    if ratio > 2.2:[m
[31m-        return "Wide"[m
[31m-    elif ratio < 1.4:[m
[31m-        return "Full"[m
[31m-    else:[m
[31m-        return "Medium"[m
[31m-[m
[31m-def analyze_nose_shape(landmarks):[m
[31m-    nose = landmarks[27:36][m
[31m-    width = np.linalg.norm(nose[0] - nose[4])[m
[31m-    height = np.linalg.norm(nose[0] - nose[6])[m
[31m-    nostril_flare = np.linalg.norm(landmarks[31] - landmarks[35])[m
[31m-    ratio = width / height if height != 0 else 0[m
[31m-    if nostril_flare / width > 0.5:[m
[31m-        return "Flared"[m
[31m-    elif ratio > 0.9:[m
[31m-        return "Wide"[m
[31m-    elif ratio < 0.6:[m
[31m-        return "Narrow"[m
[31m-    else:[m
[31m-        return "Medium"[m
[31m-[m
[31m-def analyze_eye_shape(landmarks):[m
[31m-    left_eye = landmarks[36:42][m
[31m-    right_eye = landmarks[42:48][m
[31m-    def eye_features(eye):[m
[31m-        width = np.linalg.norm(eye[0] - eye[3])[m
[31m-        height = (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / 2[m
[31m-        area = 0.5 * abs([m
[31m-            sum(eye[i][0]*eye[(i+1)%6][1] - eye[(i+1)%6][0]*eye[i][1] for i in range(6))[m
[31m-        )[m
[31m-        ratio = width / height if height != 0 else 0[m
[31m-        return ratio, area[m
[31m-    left_ratio, left_area = eye_features(left_eye)[m
[31m-    right_ratio, right_area = eye_features(right_eye)[m
[31m-    avg_ratio = (left_ratio + right_ratio) / 2[m
[31m-    avg_area = (left_area + right_area) / 2[m
[31m-    if avg_ratio > 2.5:[m
[31m-        return "Almond"[m
[31m-    elif avg_area > 400:[m
[31m-        return "Large"[m
[31m-    elif avg_area < 200:[m
[31m-        return "Small"[m
[31m-    else:[m
[31m-        return "Round"[m
[31m-[m
[31m-# %%[m
[31m-class MainMenu(QWidget):[m
[31m-    def __init__(self, ocr_callback, face_callback):[m
[31m-        super().__init__()[m
[31m-        self.setWindowTitle('Main Menu')[m
[31m-        layout = QVBoxLayout()[m
[31m-        self.ocr_btn = QPushButton('OCR Numbers')[m
[31m-        self.face_btn = QPushButton('Analyze Face Features')[m
[31m-        self.exit_btn = QPushButton('Exit')[m
[31m-        layout.addWidget(self.ocr_btn)[m
[31m-        layout.addWidget(self.face_btn)[m
[31m-        layout.addWidget(self.exit_btn)[m
[31m-        self.setLayout(layout)[m
[31m-        self.ocr_btn.clicked.connect(ocr_callback)[m
[31m-        self.face_btn.clicked.connect(face_callback)[m
[31m-        self.exit_btn.clicked.connect(self.exit_all)[m
[31m-        self.resize(400, 300)[m
[31m-[m
[31m-    def exit_all(self):[m
[31m-        QApplication.quit()[m
[31m-[m
[31m-class OCRWindow(QWidget):[m
[31m-    def __init__(self, back_callback):[m
[31m-        super().__init__()[m
[31m-        self.setWindowTitle('OCR Numbers')[m
[31m-        self.image_label = QLabel()[m
[31m-        self.result_label = QLabel('Detected Number:')[m
[31m-        self.back_btn = QPushButton('Back to Main Menu')[m
[31m-        self.back_btn.clicked.connect(back_callback)[m
[31m-        layout = QVBoxLayout()[m
[31m-        layout.addWidget(self.image_label)[m
[31m-        layout.addWidget(self.result_label)[m
[31m-        layout.addWidget(self.back_btn)[m
[31m-        self.setLayout(layout)[m
[31m-        self.cap = cv2.VideoCapture(0)[m
[31m-        self.timer = QTimer()[m
[31m-        self.timer.timeout.connect(self.update_frame)[m
[31m-        self.timer.start(30)[m
[31m-[m
[31m-    def update_frame(self):[m
[31m-        ret, frame = self.cap.read()[m
[31m-        if not ret:[m
[31m-            return[m
[31m-        frame = cv2.flip(frame, 1)[m
[31m-        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[m
[31m-        blurred = cv2.GaussianBlur(gray, (5, 5), 0)[m
[31m-        edged = cv2.Canny(blurred, 30, 150)[m
[31m-        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[m
[31m-        contours = sorted(contours, key = x_cord_contour, reverse = False)[m
[31m-        full_number = [][m
[31m-        for c in contours:[m
[31m-            (x, y, w, h) = cv2.boundingRect(c)[m
[31m-            if w >= 5 and h >= 25:[m
[31m-                roi = blurred[y:y + h, x:x + w][m
[31m-                ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)[m
[31m-                squared = makeSquare(roi)[m
[31m-                final = resize_to_pixel(20, squared)[m
[31m-                final_array = final.reshape((1,400)).astype(np.float32)[m
[31m-                ret, result, neighbours, dist = knn.findNearest(final_array, k=1)[m
[31m-                number = str(int(result[0, 0]))[m
[31m-                full_number.append(number)[m
[31m-                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)[m
[31m-                cv2.putText(frame, number, (x , y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)[m
[31m-        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[m
[31m-        h, w, ch = rgb_image.shape[m
[31m-        bytes_per_line = ch * w[m
[31m-        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)[m
[31m-        self.image_label.setPixmap(QPixmap.fromImage(qt_image))[m
[31m-        self.result_label.setText('Detected Number: ' + ''.join(full_number))[m
[31m-[m
[31m-    def closeEvent(self, event):[m
[31m-        if self.cap:[m
[31m-            self.cap.release()[m
[31m-        self.timer.stop()[m
[31m-        cv2.destroyAllWindows()[m
[31m-        event.accept()[m
[31m-[m
[31m-class FaceAnalyzerApp(QWidget):[m
[31m-    def __init__(self, back_callback=None):[m
[31m-        super().__init__()[m
[31m-        self.setWindowTitle('Face & Feature Shape Analyzer')[m
[31m-        self.image_label = QLabel()[m
[31m-        self.result_label = QLabel('Results will appear here.')[m
[31m-        self.ocr_label = QLabel('OCR:')[m
[31m-        self.cb_face_shape = QCheckBox('Show Face Shape Label')[m
[31m-        self.cb_face_shape.setChecked(True)[m
[31m-        self.cb_lip_shape = QCheckBox('Show Lip Shape Label')[m
[31m-        self.cb_lip_shape.setChecked(False)[m
[31m-        self.cb_nose_shape = QCheckBox('Show Nose Shape Label')[m
[31m-        self.cb_nose_shape.setChecked(False)[m
[31m-        self.cb_eye_shape = QCheckBox('Show Eye Shape Label')[m
[31m-        self.cb_eye_shape.setChecked(False)[m
[31m-        self.cb_landmarks = QCheckBox('Show Landmark Points')[m
[31m-        self.cb_landmarks.setChecked(False)[m
[31m-        toggles_layout = QHBoxLayout()[m
[31m-        toggles_layout.addWidget(self.cb_face_shape)[m
[31m-        toggles_layout.addWidget(self.cb_lip_shape)[m
[31m-        toggles_layout.addWidget(self.cb_nose_shape)[m
[31m-        toggles_layout.addWidget(self.cb_eye_shape)[m
[31m-        toggles_layout.addWidget(self.cb_landmarks)[m
[31m-        layout = QVBoxLayout()[m
[31m-        layout.addWidget(self.image_label)[m
[31m-        layout.addWidget(self.result_label)[m
[31m-        layout.addWidget(self.ocr_label)[m
[31m-        layout.addLayout(toggles_layout)[m
[31m-        if back_callback:[m
[31m-            self.back_btn = QPushButton('Back to Main Menu')[m
[31m-            self.back_btn.clicked.connect(back_callback)[m
[31m-            layout.addWidget(self.back_btn)[m
[31m-        self.setLayout(layout)[m
[31m-        self.resize(900, 700)[m
[31m-        self.cap = None[m
[31m-        self.timer = QTimer()[m
[31m-        self.timer.timeout.connect(self.update_frame)[m
[31m-        self.cb_face_shape.stateChanged.connect(self.update_frame)[m
[31m-        self.cb_lip_shape.stateChanged.connect(self.update_frame)[m
[31m-        self.cb_nose_shape.stateChanged.connect(self.update_frame)[m
[31m-        self.cb_eye_shape.stateChanged.connect(self.update_frame)[m
[31m-        self.cb_landmarks.stateChanged.connect(self.update_frame)[m
[31m-        self.start_webcam()[m
[31m-[m
[31m-    def start_webcam(self):[m
[31m-        if self.cap is None:[m
[31m-            self.cap = cv2.VideoCapture(0)[m
[31m-        if not self.timer.isActive():[m
[31m-            self.timer.start(30)[m
[31m-[m
[31m-    def update_frame(self):[m
[31m-        ret, frame = self.cap.read()[m
[31m-        if not ret:[m
[31m-            return[m
[31m-        frame = cv2.flip(frame, 1)[m
[31m-        orig_frame = frame.copy()[m
[31m-        gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)[m
[31m-        faces = face_detector(gray)[m
[31m-        result_text = "No face detected."[m
[31m-        ocr_text = ""[m
[31m-        for face in faces:[m
[31m-            shape = landmark_predictor(gray, face)[m
[31m-            landmarks = shape_to_np(shape)[m
[31m-            # Draw overlays for features if their label is checked[m
[31m-            if self.cb_face_shape.isChecked():[m
[31m-                # Outline jaw/face[m
[31m-                face_outline = np.array(landmarks[0:17], np.int32)[m
[31m-                cv2.polylines(frame, [face_outline], False, (0,0,255), 2)[m
[31m-                cv2.putText(frame, f"Face: {analyze_face_shape(landmarks)}", (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)[m
[31m-            if self.cb_lip_shape.isChecked():[m
[31m-                lips_outline = np.array(landmarks[48:60], np.int32)[m
[31m-                cv2.polylines(frame, [lips_outline], True, (255,0,255), 2)[m
[31m-                lip_center = tuple(np.mean(landmarks[48:68], axis=0).astype(int))[m
[31m-                cv2.putText(frame, f"Lips: {analyze_lip_shape(landmarks)}", (lip_center[0]-40, lip_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)[m
[31m-            if self.cb_nose_shape.isChecked():[m
[31m-                nose_outline = np.array(landmarks[27:36], np.int32)[m
[31m-                cv2.polylines(frame, [nose_outline], False, (0,128,255), 2)[m
[31m-                nose_center = tuple(np.mean(landmarks[27:36], axis=0).astype(int))[m
[31m-                cv2.putText(frame, f"Nose: {analyze_nose_shape(landmarks)}", (nose_center[0]-30, nose_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)[m
[31m-            if self.cb_eye_shape.isChecked():[m
[31m-                left_eye_outline = np.array(landmarks[36:42], np.int32)[m
[31m-                right_eye_outline = np.array(landmarks[42:48], np.int32)[m
[31m-                cv2.polylines(frame, [left_eye_outline], True, (0,255,255), 2)[m
[31m-                cv2.polylines(frame, [right_eye_outline], True, (0,255,255), 2)[m
[31m-                left_eye_center = tuple(np.mean(landmarks[36:42], axis=0).astype(int))[m
[31m-                right_eye_center = tuple(np.mean(landmarks[42:48], axis=0).astype(int))[m
[31m-                cv2.putText(frame, f"Eye: {analyze_eye_shape(landmarks)}", (left_eye_center[0]-20, left_eye_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)[m
[31m-                cv2.putText(frame, f"Eye: {analyze_eye_shape(landmarks)}", (right_eye_center[0]-20, right_eye_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)[m
[31m-            if self.cb_landmarks.isChecked():[m
[31m-                for (x, y) in landmarks:[m
[31m-                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)[m
[31m-            result_text = f"Face: {analyze_face_shape(landmarks)}\nLips: {analyze_lip_shape(landmarks)}\nNose: {analyze_nose_shape(landmarks)}\nEyes: {analyze_eye_shape(landmarks)}"[m
[31m-            break[m
[31m-        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[m
[31m-        h, w, ch = rgb_image.shape[m
[31m-        bytes_per_line = ch * w[m
[31m-        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)[m
[31m-        self.image_label.setPixmap(QPixmap.fromImage(qt_image))[m
[31m-        self.result_label.setText(result_text)[m
[31m-[m
[31m-    def closeEvent(self, event):[m
[31m-        if self.cap:[m
[31m-            self.cap.release()[m
[31m-        self.timer.stop()[m
[31m-        cv2.destroyAllWindows()[m
[31m-        event.accept()[m
[31m-[m
[31m-class MainController:[m
[31m-    def __init__(self):[m
[31m-        self.app = QApplication(sys.argv)[m
[31m-        self.menu = MainMenu(self.show_ocr, self.show_face)[m
[31m-        self.ocr_window = None[m
[31m-        self.face_window = None[m
[31m-        self.menu.show()[m
[31m-[m
[31m-    def show_ocr(self):[m
[31m-        if self.ocr_window is None:[m
[31m-            self.ocr_window = OCRWindow(self.show_menu)[m
[31m-        self.menu.hide()[m
[31m-        self.ocr_window.show()[m
[31m-[m
[31m-    def show_face(self):[m
[31m-        if self.face_window is None:[m
[31m-            self.face_window = FaceAnalyzerApp(self.show_menu)[m
[31m-        self.menu.hide()[m
[31m-        self.face_window.show()    def show_menu(self):[m
[31m-        if self.ocr_window:[m
[31m-            self.ocr_window.hide()[m
[31m-        if self.face_window:[m
[31m-            self.face_window.hide()[m
[31m-        self.menu.show()[m
[31m-        [m
[31m-    def run(self):[m
[31m-        sys.exit(self.app.exec_())[m
[31m-[m
[31m-if __name__ == '__main__':[m
[31m-    try:[m
[31m-        controller = MainController()[m
[31m-        controller.run()[m
[31m-    except SystemExit:[m
[31m-        # Release OpenCV resources and suppress traceback in interactive environments[m
[31m-        try:[m
[31m-            # Make sure all webcam resources are released[m
[31m-            if hasattr(controller, 'ocr_window') and controller.ocr_window and controller.ocr_window.cap:[m
[31m-                controller.ocr_window.cap.release()[m
[31m-            if hasattr(controller, 'face_window') and controller.face_window and controller.face_window.cap:[m
[31m-                controller.face_window.cap.release()[m
[31m-            cv2.destroyAllWindows()[m
[31m-        except Exception:[m
[31m-            pass[m
\ No newline at end of file[m
[1mdiff --git a/new_CMSC191_Salcedo_Final Project_cells.py b/new_CMSC191_Salcedo_Final Project_cells.py[m
[1mnew file mode 100644[m
[1mindex 0000000..4899658[m
[1m--- /dev/null[m
[1m+++ b/new_CMSC191_Salcedo_Final Project_cells.py[m	
[36m@@ -0,0 +1,409 @@[m
[32m+[m[32m# %% [markdown][m
[32m+[m[32m# CMSC191 Final Project: Face and Feature Shape Analyzer[m
[32m+[m[32m#[m
[32m+[m[32m# Dependencies:[m
[32m+[m[32m# - opencv-python[m
[32m+[m[32m# - pyqt5[m
[32m+[m[32m# - numpy[m
[32m+[m[32m# - dlib (for facial landmark detection)[m
[32m+[m[32m#[m
[32m+[m[32m# Before running, install dependencies:[m
[32m+[m[32m# pip install opencv-python pyqt5 numpy dlib[m
[32m+[m
[32m+[m[32m# %%[m
[32m+[m[32mimport sys[m
[32m+[m[32mimport os[m
[32m+[m[32mimport cv2[m
[32m+[m[32mimport numpy as np[m
[32m+[m[32mfrom PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QCheckBox, QHBoxLayout[m
[32m+[m[32mfrom PyQt5.QtGui import QImage, QPixmap[m
[32m+[m[32mfrom PyQt5.QtCore import QTimer[m
[32m+[m[32mimport dlib[m
[32m+[m
[32m+[m[32m# %%[m
[32m+[m[32m# Load dlib's pre-trained face detector and shape predictor[m
[32m+[m[32mface_detector = dlib.get_frontal_face_detector()[m
[32m+[m[32mscript_dir = os.path.dirname(os.path.abspath(__file__))[m
[32m+[m[32mmodel_path = os.path.join(script_dir, 'shape_predictor_68_face_landmarks.dat')[m
[32m+[m[32mlandmark_predictor = dlib.shape_predictor(model_path)[m
[32m+[m
[32m+[m[32m# --- KNN Digit Recognizer Setup ---[m
[32m+[m[32mdigits_img = cv2.imread(os.path.join(script_dir, 'cnn', 'digits.png'))[m
[32m+[m[32mgray_digits = cv2.cvtColor(digits_img, cv2.COLOR_BGR2GRAY)[m
[32m+[m[32mcells = [np.hsplit(row,100) for row in np.vsplit(gray_digits,50)][m
[32m+[m[32mx = np.array(cells)[m
[32m+[m[32mtrain = x[:,:70].reshape(-1,400).astype(np.float32)[m
[32m+[m[32mtrain_labels = np.repeat(np.arange(10),350)[:,np.newaxis][m
[32m+[m[32mknn = cv2.ml.KNearest_create()[m
[32m+[m[32mknn.train(train, cv2.ml.ROW_SAMPLE, train_labels)[m
[32m+[m[32m# --- End KNN Setup ---[m
[32m+[m
[32m+[m[32m# Helper functions for digit ROI extraction[m
[32m+[m
[32m+[m[32mdef x_cord_contour(contour):[m
[32m+[m[32m    # Returns the x coordinate of the bounding rectangle of a contour[m
[32m+[m[32m    M = cv2.moments(contour)[m
[32m+[m[32m    if M['m00'] != 0:[m
[32m+[m[32m        return int(M['m10']/M['m00'])[m
[32m+[m[32m    else:[m
[32m+[m[32m        return 0[m
[32m+[m
[32m+[m[32mdef makeSquare(not_square):[m
[32m+[m[32m    BLACK = [0,0,0][m
[32m+[m[32m    img_dim = not_square.shape[m
[32m+[m[32m    height = img_dim[0][m
[32m+[m[32m    width = img_dim[1][m
[32m+[m[32m    if (height == width):[m
[32m+[m[32m        square = not_square[m
[32m+[m[32m        return square[m
[32m+[m[32m    else:[m
[32m+[m[32m        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)[m
[32m+[m[32m        height = height * 2[m
[32m+[m[32m        width = width * 2[m
[32m+[m[32m        if (height > width):[m
[32m+[m[32m            pad = int((height - width)/2)[m
[32m+[m[32m            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,pad,cv2.BORDER_CONSTANT,value=BLACK)[m
[32m+[m[32m        else:[m
[32m+[m[32m            pad = int((width - height)/2)[m
[32m+[m[32m            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,cv2.BORDER_CONSTANT,value=BLACK)[m
[32m+[m[32m        return doublesize_square[m
[32m+[m
[32m+[m[32mdef resize_to_pixel(dimensions, image):[m
[32m+[m[32m    buffer_pix = 4[m
[32m+[m[32m    dimensions  = dimensions - buffer_pix[m
[32m+[m[32m    squared = image[m
[32m+[m[32m    r = float(dimensions) / squared.shape[1][m
[32m+[m[32m    dim = (dimensions, int(squared.shape[0] * r))[m
[32m+[m[32m    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)[m
[32m+[m[32m    img_dim2 = resized.shape[m
[32m+[m[32m    height_r = img_dim2[0][m
[32m+[m[32m    width_r = img_dim2[1][m
[32m+[m[32m    BLACK = [0,0,0][m
[32m+[m[32m    if (height_r > width_r):[m
[32m+[m[32m        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)[m
[32m+[m[32m    if (height_r < width_r):[m
[32m+[m[32m        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)[m
[32m+[m[32m    p = 2[m
[32m+[m[32m    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)[m
[32m+[m[32m    return ReSizedImg[m
[32m+[m
[32m+[m[32m# %%[m
[32m+[m[32m# Helper function to convert dlib shape to numpy array[m
[32m+[m[32mdef shape_to_np(shape, dtype="int"):[m
[32m+[m[32m    coords = np.zeros((68, 2), dtype=dtype)[m
[32m+[m[32m    for i in range(0, 68):[m
[32m+[m[32m        coords[i] = (shape.part(i).x, shape.part(i).y)[m
[32m+[m[32m    return coords[m
[32m+[m
[32m+[m[32m# %%[m
[32m+[m[32mdef analyze_face_shape(landmarks):[m
[32m+[m[32m    # jaw width (points 4 to 12)[m
[32m+[m[32m    jaw_width = np.linalg.norm(landmarks[4] - landmarks[12])[m
[32m+[m[32m    # cheekbone width (points 1 to 15)[m
[32m+[m[32m    cheekbone_width = np.linalg.norm(landmarks[1] - landmarks[15])[m
[32m+[m[32m    # forehead width (approximate: points 17 to 26)[m
[32m+[m[32m    forehead_width = np.linalg.norm(landmarks[17] - landmarks[26])[m
[32m+[m[32m    # face height (chin to midpoint between eyebrows)[m
[32m+[m[32m    chin = landmarks[8][m
[32m+[m[32m    brow_mid = ((landmarks[21] + landmarks[22]) // 2)[m
[32m+[m[32m    face_height = np.linalg.norm(chin - brow_mid)[m
[32m+[m[32m    # ratios[m
[32m+[m[32m    jaw_to_height = jaw_width / face_height[m
[32m+[m[32m    cheek_to_height = cheekbone_width / face_height[m
[32m+[m[32m    forehead_to_height = forehead_width / face_height[m
[32m+[m[32m    # heuristic rules for face shape[m
[32m+[m[32m    if abs(jaw_to_height - cheek_to_height) < 0.05 and abs(cheek_to_height - forehead_to_height) < 0.05:[m
[32m+[m[32m        return "Square"[m
[32m+[m[32m    elif cheek_to_height > jaw_to_height and cheek_to_height > forehead_to_height:[m
[32m+[m[32m        return "Diamond"[m
[32m+[m[32m    elif jaw_to_height > cheek_to_height and jaw_to_height > forehead_to_height:[m
[32m+[m[32m        return "Triangle"[m
[32m+[m[32m    elif forehead_to_height > cheek_to_height and forehead_to_height > jaw_to_height:[m
[32m+[m[32m        return "Heart"[m
[32m+[m[32m    elif face_height / cheekbone_width > 1.5:[m
[32m+[m[32m        return "Oval"[m
[32m+[m[32m    else:[m
[32m+[m[32m        return "Round"[m
[32m+[m
[32m+[m[32mdef analyze_lip_shape(landmarks):[m
[32m+[m[32m    lips = landmarks[48:68][m
[32m+[m[32m    width = np.linalg.norm(lips[0] - lips[6])[m
[32m+[m[32m    upper_height = np.linalg.norm(lips[3] - lips[13])[m
[32m+[m[32m    lower_height = np.linalg.norm(lips[9] - lips[19])[m
[32m+[m[32m    avg_height = (upper_height + lower_height) / 2[m
[32m+[m[32m    ratio = width / avg_height if avg_height != 0 else 0[m
[32m+[m[32m    if ratio > 2.2:[m
[32m+[m[32m        return "Wide"[m
[32m+[m[32m    elif ratio < 1.4:[m
[32m+[m[32m        return "Full"[m
[32m+[m[32m    else:[m
[32m+[m[32m        return "Medium"[m
[32m+[m
[32m+[m[32mdef analyze_nose_shape(landmarks):[m
[32m+[m[32m    nose = landmarks[27:36][m
[32m+[m[32m    width = np.linalg.norm(nose[0] - nose[4])[m
[32m+[m[32m    height = np.linalg.norm(nose[0] - nose[6])[m
[32m+[m[32m    nostril_flare = np.linalg.norm(landmarks[31] - landmarks[35])[m
[32m+[m[32m    ratio = width / height if height != 0 else 0[m
[32m+[m[32m    if nostril_flare / width > 0.5:[m
[32m+[m[32m        return "Flared"[m
[32m+[m[32m    elif ratio > 0.9:[m
[32m+[m[32m        return "Wide"[m
[32m+[m[32m    elif ratio < 0.6:[m
[32m+[m[32m        return "Narrow"[m
[32m+[m[32m    else:[m
[32m+[m[32m        return "Medium"[m
[32m+[m
[32m+[m[32mdef analyze_eye_shape(landmarks):[m
[32m+[m[32m    left_eye = landmarks[36:42][m
[32m+[m[32m    right_eye = landmarks[42:48][m
[32m+[m[32m    def eye_features(eye):[m
[32m+[m[32m        width = np.linalg.norm(eye[0] - eye[3])[m
[32m+[m[32m        height = (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / 2[m
[32m+[m[32m        area = 0.5 * abs([m
[32m+[m[32m            sum(eye[i][0]*eye[(i+1)%6][1] - eye[(i+1)%6][0]*eye[i][1] for i in range(6))[m
[32m+[m[32m        )[m
[32m+[m[32m        ratio = width / height if height != 0 else 0[m
[32m+[m[32m        return ratio, area[m
[32m+[m[32m    left_ratio, left_area = eye_features(left_eye)[m
[32m+[m[32m    right_ratio, right_area = eye_features(right_eye)[m
[32m+[m[32m    avg_ratio = (left_ratio + right_ratio) / 2[m
[32m+[m[32m    avg_area = (left_area + right_area) / 2[m
[32m+[m[32m    if avg_ratio > 2.5:[m
[32m+[m[32m        return "Almond"[m
[32m+[m[32m    elif avg_area > 400:[m
[32m+[m[32m        return "Large"[m
[32m+[m[32m    elif avg_area < 200:[m
[32m+[m[32m        return "Small"[m
[32m+[m[32m    else:[m
[32m+[m[32m        return "Round"[m
[32m+[m
[32m+[m[32m# %%[m
[32m+[m[32mclass MainMenu(QWidget):[m
[32m+[m[32m    def __init__(self, ocr_callback, face_callback):[m
[32m+[m[32m        super().__init__()[m
[32m+[m[32m        self.setWindowTitle('Main Menu')[m
[32m+[m[32m        layout = QVBoxLayout()[m
[32m+[m[32m        self.ocr_btn = QPushButton('OCR Numbers')[m
[32m+[m[32m        self.face_btn = QPushButton('Analyze Face Features')[m
[32m+[m[32m        self.exit_btn = QPushButton('Exit')[m
[32m+[m[32m        layout.addWidget(self.ocr_btn)[m
[32m+[m[32m        layout.addWidget(self.face_btn)[m
[32m+[m[32m        layout.addWidget(self.exit_btn)[m
[32m+[m[32m        self.setLayout(layout)[m
[32m+[m[32m        self.ocr_btn.clicked.connect(ocr_callback)[m
[32m+[m[32m        self.face_btn.clicked.connect(face_callback)[m
[32m+[m[32m        self.exit_btn.clicked.connect(self.exit_all)[m
[32m+[m[32m        self.resize(400, 300)[m
[32m+[m
[32m+[m[32m    def exit_all(self):[m
[32m+[m[32m        QApplication.quit()[m
[32m+[m
[32m+[m[32mclass OCRWindow(QWidget):[m
[32m+[m[32m    def __init__(self, back_callback):[m
[32m+[m[32m        super().__init__()[m
[32m+[m[32m        self.setWindowTitle('OCR Numbers')[m
[32m+[m[32m        self.image_label = QLabel()[m
[32m+[m[32m        self.result_label = QLabel('Detected Number:')[m
[32m+[m[32m        self.back_btn = QPushButton('Back to Main Menu')[m
[32m+[m[32m        self.back_btn.clicked.connect(back_callback)[m
[32m+[m[32m        layout = QVBoxLayout()[m
[32m+[m[32m        layout.addWidget(self.image_label)[m
[32m+[m[32m        layout.addWidget(self.result_label)[m
[32m+[m[32m        layout.addWidget(self.back_btn)[m
[32m+[m[32m        self.setLayout(layout)[m
[32m+[m[32m        self.cap = cv2.VideoCapture(0)[m
[32m+[m[32m        self.timer = QTimer()[m
[32m+[m[32m        self.timer.timeout.connect(self.update_frame)[m
[32m+[m[32m        self.timer.start(30)[m
[32m+[m
[32m+[m[32m    def update_frame(self):[m
[32m+[m[32m        ret, frame = self.cap.read()[m
[32m+[m[32m        if not ret:[m
[32m+[m[32m            return[m
[32m+[m[32m        frame = cv2.flip(frame, 1)[m
[32m+[m[32m        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[m
[32m+[m[32m        blurred = cv2.GaussianBlur(gray, (5, 5), 0)[m
[32m+[m[32m        edged = cv2.Canny(blurred, 30, 150)[m
[32m+[m[32m        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[m
[32m+[m[32m        contours = sorted(contours, key = x_cord_contour, reverse = False)[m
[32m+[m[32m        full_number = [][m
[32m+[m[32m        for c in contours:[m
[32m+[m[32m            (x, y, w, h) = cv2.boundingRect(c)[m
[32m+[m[32m            if w >= 5 and h >= 25:[m
[32m+[m[32m                roi = blurred[y:y + h, x:x + w][m
[32m+[m[32m                ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)[m
[32m+[m[32m                squared = makeSquare(roi)[m
[32m+[m[32m                final = resize_to_pixel(20, squared)[m
[32m+[m[32m                final_array = final.reshape((1,400)).astype(np.float32)[m
[32m+[m[32m                ret, result, neighbours, dist = knn.findNearest(final_array, k=1)[m
[32m+[m[32m                number = str(int(result[0, 0]))[m
[32m+[m[32m                full_number.append(number)[m
[32m+[m[32m                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)[m
[32m+[m[32m                cv2.putText(frame, number, (x , y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)[m
[32m+[m[32m        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[m
[32m+[m[32m        h, w, ch = rgb_image.shape[m
[32m+[m[32m        bytes_per_line = ch * w[m
[32m+[m[32m        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)[m
[32m+[m[32m        self.image_label.setPixmap(QPixmap.fromImage(qt_image))[m
[32m+[m[32m        self.result_label.setText('Detected Number: ' + ''.join(full_number))[m
[32m+[m
[32m+[m[32m    def closeEvent(self, event):[m
[32m+[m[32m        if self.cap:[m
[32m+[m[32m            self.cap.release()[m
[32m+[m[32m        self.timer.stop()[m
[32m+[m[32m        cv2.destroyAllWindows()[m
[32m+[m[32m        event.accept()[m
[32m+[m
[32m+[m[32mclass FaceAnalyzerApp(QWidget):[m
[32m+[m[32m    def __init__(self, back_callback=None):[m
[32m+[m[32m        super().__init__()[m
[32m+[m[32m        self.setWindowTitle('Face & Feature Shape Analyzer')[m
[32m+[m[32m        self.image_label = QLabel()[m
[32m+[m[32m        self.result_label = QLabel('Results will appear here.')[m
[32m+[m[32m        self.ocr_label = QLabel('OCR:')[m
[32m+[m[32m        self.cb_face_shape = QCheckBox('Show Face Shape Label')[m
[32m+[m[32m        self.cb_face_shape.setChecked(True)[m
[32m+[m[32m        self.cb_lip_shape = QCheckBox('Show Lip Shape Label')[m
[32m+[m[32m        self.cb_lip_shape.setChecked(False)[m
[32m+[m[32m        self.cb_nose_shape = QCheckBox('Show Nose Shape Label')[m
[32m+[m[32m        self.cb_nose_shape.setChecked(False)[m
[32m+[m[32m        self.cb_eye_shape = QCheckBox('Show Eye Shape Label')[m
[32m+[m[32m        self.cb_eye_shape.setChecked(False)[m
[32m+[m[32m        self.cb_landmarks = QCheckBox('Show Landmark Points')[m
[32m+[m[32m        self.cb_landmarks.setChecked(False)[m
[32m+[m[32m        toggles_layout = QHBoxLayout()[m
[32m+[m[32m        toggles_layout.addWidget(self.cb_face_shape)[m
[32m+[m[32m        toggles_layout.addWidget(self.cb_lip_shape)[m
[32m+[m[32m        toggles_layout.addWidget(self.cb_nose_shape)[m
[32m+[m[32m        toggles_layout.addWidget(self.cb_eye_shape)[m
[32m+[m[32m        toggles_layout.addWidget(self.cb_landmarks)[m
[32m+[m[32m        layout = QVBoxLayout()[m
[32m+[m[32m        layout.addWidget(self.image_label)[m
[32m+[m[32m        layout.addWidget(self.result_label)[m
[32m+[m[32m        layout.addWidget(self.ocr_label)[m
[32m+[m[32m        layout.addLayout(toggles_layout)[m
[32m+[m[32m        if back_callback:[m
[32m+[m[32m            self.back_btn = QPushButton('Back to Main Menu')[m
[32m+[m[32m            self.back_btn.clicked.connect(back_callback)[m
[32m+[m[32m            layout.addWidget(self.back_btn)[m
[32m+[m[32m        self.setLayout(layout)[m
[32m+[m[32m        self.resize(900, 700)[m
[32m+[m[32m        self.cap = None[m
[32m+[m[32m        self.timer = QTimer()[m
[32m+[m[32m        self.timer.timeout.connect(self.update_frame)[m
[32m+[m[32m        self.cb_face_shape.stateChanged.connect(self.update_frame)[m
[32m+[m[32m        self.cb_lip_shape.stateChanged.connect(self.update_frame)[m
[32m+[m[32m        self.cb_nose_shape.stateChanged.connect(self.update_frame)[m
[32m+[m[32m        self.cb_eye_shape.stateChanged.connect(self.update_frame)[m
[32m+[m[32m        self.cb_landmarks.stateChanged.connect(self.update_frame)[m
[32m+[m[32m        self.start_webcam()[m
[32m+[m
[32m+[m[32m    def start_webcam(self):[m
[32m+[m[32m        if self.cap is None:[m
[32m+[m[32m            self.cap = cv2.VideoCapture(0)[m
[32m+[m[32m        if not self.timer.isActive():[m
[32m+[m[32m            self.timer.start(30)[m
[32m+[m
[32m+[m[32m    def update_frame(self):[m
[32m+[m[32m        ret, frame = self.cap.read()[m
[32m+[m[32m        if not ret:[m
[32m+[m[32m            return[m
[32m+[m[32m        frame = cv2.flip(frame, 1)[m
[32m+[m[32m        orig_frame = frame.copy()[m
[32m+[m[32m        gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)[m
[32m+[m[32m        faces = face_detector(gray)[m
[32m+[m[32m        result_text = "No face detected."[m
[32m+[m[32m        ocr_text = ""[m
[32m+[m[32m        for face in faces:[m
[32m+[m[32m            shape = landmark_predictor(gray, face)[m
[32m+[m[32m            landmarks = shape_to_np(shape)[m
[32m+[m[32m            # Draw overlays for features if their label is checked[m
[32m+[m[32m            if self.cb_face_shape.isChecked():[m
[32m+[m[32m                # Outline jaw/face[m
[32m+[m[32m                face_outline = np.array(landmarks[0:17], np.int32)[m
[32m+[m[32m                cv2.polylines(frame, [face_outline], False, (0,0,255), 2)[m
[32m+[m[32m                cv2.putText(frame, f"Face: {analyze_face_shape(landmarks)}", (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)[m
[32m+[m[32m            if self.cb_lip_shape.isChecked():[m
[32m+[m[32m                lips_outline = np.array(landmarks[48:60], np.int32)[m
[32m+[m[32m                cv2.polylines(frame, [lips_outline], True, (255,0,255), 2)[m
[32m+[m[32m                lip_center = tuple(np.mean(landmarks[48:68], axis=0).astype(int))[m
[32m+[m[32m                cv2.putText(frame, f"Lips: {analyze_lip_shape(landmarks)}", (lip_center[0]-40, lip_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)[m
[32m+[m[32m            if self.cb_nose_shape.isChecked():[m
[32m+[m[32m                nose_outline = np.array(landmarks[27:36], np.int32)[m
[32m+[m[32m                cv2.polylines(frame, [nose_outline], False, (0,128,255), 2)[m
[32m+[m[32m                nose_center = tuple(np.mean(landmarks[27:36], axis=0).astype(int))[m
[32m+[m[32m                cv2.putText(frame, f"Nose: {analyze_nose_shape(landmarks)}", (nose_center[0]-30, nose_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)[m
[32m+[m[32m            if self.cb_eye_shape.isChecked():[m
[32m+[m[32m                left_eye_outline = np.array(landmarks[36:42], np.int32)[m
[32m+[m[32m                right_eye_outline = np.array(landmarks[42:48], np.int32)[m
[32m+[m[32m                cv2.polylines(frame, [left_eye_outline], True, (0,255,255), 2)[m
[32m+[m[32m                cv2.polylines(frame, [right_eye_outline], True, (0,255,255), 2)[m
[32m+[m[32m                left_eye_center = tuple(np.mean(landmarks[36:42], axis=0).astype(int))[m
[32m+[m[32m                right_eye_center = tuple(np.mean(landmarks[42:48], axis=0).astype(int))[m
[32m+[m[32m                cv2.putText(frame, f"Eye: {analyze_eye_shape(landmarks)}", (left_eye_center[0]-20, left_eye_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)[m
[32m+[m[32m                cv2.putText(frame, f"Eye: {analyze_eye_shape(landmarks)}", (right_eye_center[0]-20, right_eye_center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)[m
[32m+[m[32m            if self.cb_landmarks.isChecked():[m
[32m+[m[32m                for (x, y) in landmarks:[m
[32m+[m[32m                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)[m
[32m+[m[32m            result_text = f"Face: {analyze_face_shape(landmarks)}\nLips: {analyze_lip_shape(landmarks)}\nNose: {analyze_nose_shape(landmarks)}\nEyes: {analyze_eye_shape(landmarks)}"[m
[32m+[m[32m            break[m
[32m+[m[32m        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[m
[32m+[m[32m        h, w, ch = rgb_image.shape[m
[32m+[m[32m        bytes_per_line = ch * w[m
[32m+[m[32m        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)[m
[32m+[m[32m        self.image_label.setPixmap(QPixmap.fromImage(qt_image))[m
[32m+[m[32m        self.result_label.setText(result_text)[m
[32m+[m
[32m+[m[32m    def closeEvent(self, event):[m
[32m+[m[32m        if self.cap:[m
[32m+[m[32m            self.cap.release()[m
[32m+[m[32m        self.timer.stop()[m
[32m+[m[32m        cv2.destroyAllWindows()[m
[32m+[m[32m        event.accept()[m
[32m+[m
[32m+[m[32mclass MainController:[m
[32m+[m[32m    def __init__(self):[m
[32m+[m[32m        self.app = QApplication(sys.argv)[m
[32m+[m[32m        self.menu = MainMenu(self.show_ocr, self.show_face)[m
[32m+[m[32m        self.ocr_window = None[m
[32m+[m[32m        self.face_window = None[m
[32m+[m[32m        self.menu.show()[m
[32m+[m
[32m+[m[32m    def show_ocr(self):[m
[32m+[m[32m        if self.ocr_window is None:[m
[32m+[m[32m            self.ocr_window = OCRWindow(self.show_menu)[m
[32m+[m[32m        self.menu.hide()[m
[32m+[m[32m        self.ocr_window.show()[m
[32m+[m
[32m+[m[32m    def show_face(self):[m
[32m+[m[32m        if self.face_window is None:[m
[32m+[m[32m            self.face_window = FaceAnalyzerApp(self.show_menu)[m
[32m+[m[32m        self.menu.hide()[m
[32m+[m[32m        self.face_window.show()[m
[32m+[m[41m        [m
[32m+[m[32m    def show_menu(self):[m
[32m+[m[32m        if self.ocr_window:[m
[32m+[m[32m            self.ocr_window.hide()[m
[32m+[m[32m        if self.face_window:[m
[32m+[m[32m            self.face_window.hide()[m
[32m+[m[32m        self.menu.show()[m
[32m+[m[41m        [m
[32m+[m[32m    def run(self):[m
[32m+[m[32m        sys.exit(self.app.exec_())[m
[32m+[m
[32m+[m[32mif __name__ == '__main__':[m
[32m+[m[32m    try:[m
[32m+[m[32m        controller = MainController()[m
[32m+[m[32m        controller.run()[m
[32m+[m[32m    except SystemExit:[m
[32m+[m[32m        # Release OpenCV resources and suppress traceback in interactive environments[m
[32m+[m[32m        try:[m
[32m+[m[32m            # Make sure all webcam resources are released[m
[32m+[m[32m            if hasattr(controller, 'ocr_window') and controller.ocr_window and hasattr(controller.ocr_window, 'cap') and controller.ocr_window.cap:[m
[32m+[m[32m                controller.ocr_window.cap.release()[m
[32m+[m[32m            if hasattr(controller, 'face_window') and controller.face_window and hasattr(controller.face_window, 'cap') and controller.face_window.cap:[m
[32m+[m[32m                controller.face_window.cap.release()[m
[32m+[m[32m            cv2.destroyAllWindows()[m
[32m+[m[32m        except Exception:[m
[32m+[m[32m            pass[m
