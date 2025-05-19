# chris samuel salcedo
# 2022-05055
# cmsc191 final project: face, feature shape analyzer
#
# dependencies:
# - opencv-python
# - pyqt5
# - numpy
# - dlib (for facial landmark detection)
# - tensorflow
#
# before running, install dependencies:
# pip install opencv-python pyqt5 numpy dlib tensorflow

import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QCheckBox, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QTimer, Qt
import dlib

# load dlib's pre-trained face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'shape_predictor_68_face_landmarks.dat')
landmark_predictor = dlib.shape_predictor(model_path)

# load emotion detection model
emotion_model_path = os.path.join(script_dir, 'emotions_model.hdf5')
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# load emoji pngs for AR overlay (one per emotion)
emoji_files = {
    'Angry': 'emoji_angry.png',
    'Disgust': 'emoji_disgust.png',
    'Fear': 'emoji_fear.png',
    'Happy': 'emoji_happy.png',
    'Sad': 'emoji_sad.png',
    'Surprise': 'emoji_surprise.png',
    'Neutral': 'emoji_neutral.png',
}
emoji_imgs = {}
for label, fname in emoji_files.items():
    path = os.path.join(script_dir, fname)
    if os.path.exists(path):
        emoji_imgs[label] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        emoji_imgs[label] = None

# helper function to convert dlib shape to numpy array
# returns 68x2 array of landmark coordinates

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# analyze face shape using facial landmarks
# returns a string label for face shape

def analyze_face_shape(landmarks):
    # Based on Xie & Lam (2015) and Sforza et al. (2010)
    # Calculate key facial measurements
    # Jaw width (points 4 to 12)
    jaw_width = np.linalg.norm(landmarks[4] - landmarks[12])
    # Cheekbone width (points 1 to 15)
    cheekbone_width = np.linalg.norm(landmarks[1] - landmarks[15])
    # Forehead width (points 17 to 26)
    forehead_width = np.linalg.norm(landmarks[17] - landmarks[26])
    # Face height (chin to midpoint between eyebrows)
    chin = landmarks[8]
    brow_mid = ((landmarks[21] + landmarks[22]) // 2)
    face_height = np.linalg.norm(chin - brow_mid)
    
    # Calculate ratios based on anthropometric studies
    jaw_to_height = jaw_width / face_height
    cheek_to_height = cheekbone_width / face_height
    forehead_to_height = forehead_width / face_height
    
    # Calculate facial thirds (based on Sforza et al., 2010)
    upper_third = np.linalg.norm(landmarks[19] - landmarks[27])  # hairline to glabella
    middle_third = np.linalg.norm(landmarks[27] - landmarks[33])  # glabella to subnasale
    lower_third = np.linalg.norm(landmarks[33] - landmarks[8])   # subnasale to menton
    
    # Calculate facial width ratios (based on Xie & Lam, 2015)
    bizygomatic_to_height = cheekbone_width / face_height
    bigonial_to_height = jaw_width / face_height
    
    # Initialize scores dictionary
    scores = {}
    
    # Square face shape criteria (Xie & Lam, 2015)
    # Square faces have similar width measurements and strong jawline
    square_score = 1.0 - abs(jaw_to_height - cheek_to_height) * 2.0
    if abs(jaw_to_height - 0.85) < 0.1:  # Ideal jaw-to-height ratio
        square_score += 0.2
    scores['Square'] = square_score
    
    # Diamond face shape criteria (Sforza et al., 2010)
    # Diamond faces have prominent cheekbones and narrower jaw/forehead
    diamond_score = (cheek_to_height - max(jaw_to_height, forehead_to_height)) * 2.0
    if cheek_to_height > 0.9:  # Prominent cheekbones
        diamond_score += 0.15
    scores['Diamond'] = diamond_score
    
    # Triangle face shape criteria
    # Triangle faces have wider jaw and narrower forehead
    triangle_score = (jaw_to_height - forehead_to_height) * 2.0
    if jaw_to_height > 0.9:  # Strong jawline
        triangle_score += 0.15
    scores['Triangle'] = triangle_score
    
    # Heart face shape criteria (Sforza et al., 2010)
    # Heart faces have wider forehead and narrower jaw
    heart_score = (forehead_to_height - jaw_to_height) * 2.0
    if forehead_to_height > 0.9:  # Wide forehead
        heart_score += 0.15
    scores['Heart'] = heart_score
    
    # Oval face shape criteria (Xie & Lam, 2015)
    # Oval faces have balanced proportions and slightly longer than wide
    oval_ratio = face_height / cheekbone_width
    oval_score = 0.9 - abs(oval_ratio - 1.5) * 2.0  # Ideal ratio around 1.5
    if 1.4 <= oval_ratio <= 1.6:  # Ideal range
        oval_score += 0.2
    scores['Oval'] = oval_score
    
    # Round face shape criteria (Sforza et al., 2010)
    # Round faces have similar width and height measurements
    round_ratio = face_height / cheekbone_width
    if 0.9 <= round_ratio <= 1.1:  # Nearly equal width and height
        round_score = 0.8 - abs(round_ratio - 1.0) * 4.0
    else:
        round_score = -abs(round_ratio - 1.0) * 4.0
    scores['Round'] = round_score
    
    # Find the best matching shape
    best_shape = max(scores, key=scores.get)
    best_score = scores[best_shape]
    
    # Check for borderline cases
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1 and abs(sorted_scores[0][1] - sorted_scores[1][1]) < 0.1:
        return f"{best_shape} (borderline with {sorted_scores[1][0]})"
    
    # Calculate confidence percentage
    confidence = min(max((best_score + 1) / 2, 0), 1) * 100
    return f"{best_shape} ({confidence:.0f}%)"

# analyze lip shape using landmarks
# returns a string label for lip shape

def analyze_lip_shape(landmarks):
    # Based on Ferrario et al. (1997) - Three-dimensional analysis of lip form
    lips = landmarks[48:68]
    
    # Calculate key measurements
    width = np.linalg.norm(lips[0] - lips[6])  # Total lip width
    upper_height = np.linalg.norm(lips[3] - lips[13])  # Upper lip height
    lower_height = np.linalg.norm(lips[9] - lips[19])  # Lower lip height
    cupid_width = np.linalg.norm(lips[2] - lips[4])  # Cupid's bow width
    
    # Calculate ratios
    avg_height = (upper_height + lower_height) * 0.5
    width_to_height = width / avg_height if avg_height > 0 else 0
    cupid_ratio = cupid_width / width if width > 0 else 0
    
    # Initialize scores
    scores = {}
    
    # Wide lips criteria (adjusted)
    if width_to_height > 3.0:  # Increased threshold
        scores['Wide'] = 0.7  # Reduced base score
    else:
        scores['Wide'] = (width_to_height - 2.5) * 0.8  # Reduced scaling
    
    # Full lips criteria (adjusted)
    height_to_width = avg_height / width if width > 0 else 0
    scores['Full'] = height_to_width * 3.0  # Increased weight
    if cupid_ratio > 0.4:
        scores['Full'] += 0.3
    
    # Medium lips criteria (adjusted)
    if 1.8 <= width_to_height <= 2.8:  # Widened range
        scores['Medium'] = 0.8 - abs(width_to_height - 2.3) * 1.0
        if 0.3 <= cupid_ratio <= 0.5:
            scores['Medium'] += 0.2
    else:
        scores['Medium'] = -abs(width_to_height - 2.3) * 0.8
    
    # Normalize scores
    max_score = max(scores.values())
    if max_score > 0:
        for key in scores:
            scores[key] = scores[key] / max_score
    
    # Find best shape
    best_shape = max(scores, key=scores.get)
    best_score = scores[best_shape]
    
    # Check for borderline cases
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1 and abs(sorted_scores[0][1] - sorted_scores[1][1]) < 0.15:
        return f"{best_shape} (borderline with {sorted_scores[1][0]})"
    
    # Calculate confidence
    confidence = min(max(best_score * 100, 0), 100)
    return f"{best_shape} ({confidence:.0f}%)"

# analyze nose shape using landmarks
# returns a string label for nose shape

def analyze_nose_shape(landmarks):
    # Based on Borman & Campbell (2010) - Anthropometric analysis of the human nose
    nose = landmarks[27:36]
    
    # Calculate key measurements
    # Nasal width (distance between alae)
    width = np.linalg.norm(nose[0] - nose[4])
    
    # Nasal height (from nasion to subnasale)
    height = np.linalg.norm(nose[0] - nose[6])
    
    # Nostril flare (distance between alar bases)
    nostril_flare = np.linalg.norm(landmarks[31] - landmarks[35])
    
    # Nasal bridge length (from nasion to rhinion)
    bridge_length = np.linalg.norm(nose[0] - nose[3])
    
    # Calculate ratios and proportions
    width_to_height = width / height if height != 0 else 0
    flare_ratio = nostril_flare / width
    bridge_ratio = bridge_length / height
    
    # Initialize scores dictionary
    scores = {}
    
    # Flared nose criteria (Borman & Campbell, 2010)
    # Flared noses have wider nostril flare relative to nasal width
    flared_score = (flare_ratio - 0.55) * 3.0
    if flare_ratio > 0.6:  # Significant flare
        flared_score += 0.2
    scores['Flared'] = flared_score
    
    # Wide nose criteria
    # Wide noses have larger width-to-height ratio
    wide_score = (width_to_height - 0.85) * 2.0
    if width_to_height > 0.9:  # Significantly wide
        wide_score += 0.15
    scores['Wide'] = wide_score
    
    # Narrow nose criteria
    # Narrow noses have smaller width-to-height ratio
    narrow_score = (0.65 - width_to_height) * 2.0
    if width_to_height < 0.7:  # Significantly narrow
        narrow_score += 0.15
    scores['Narrow'] = narrow_score
    
    # Medium nose criteria
    # Medium noses have balanced proportions
    if 0.75 <= width_to_height <= 0.85:
        medium_score = 0.8 - abs(width_to_height - 0.8) * 4.0
        if 0.4 <= bridge_ratio <= 0.6:  # Moderate bridge length
            medium_score += 0.15
    else:
        medium_score = -abs(width_to_height - 0.8) * 3.0
    scores['Medium'] = medium_score
    
    # Find the best matching shape
    best_shape = max(scores, key=scores.get)
    best_score = scores[best_shape]
    
    # Check for borderline cases
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1 and abs(sorted_scores[0][1] - sorted_scores[1][1]) < 0.1:
        return f"{best_shape} (borderline with {sorted_scores[1][0]})"
    
    # Calculate confidence percentage
    confidence = min(max((best_score + 1) / 2, 0), 1) * 100
    return f"{best_shape} ({confidence:.0f}%)"

# analyze eye shape using landmarks
# returns a string label for eye shape

def analyze_eye_shape(landmarks):
    # Based on Ercan et al. (2008) and Xie & Lam (2015)
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    
    def eye_features(eye):
        # Calculate key measurements for a single eye
        # Eye width (horizontal distance)
        width = np.linalg.norm(eye[0] - eye[3])
        
        # Eye height (average of upper and lower lid distances)
        upper_height = np.linalg.norm(eye[1] - eye[5])
        lower_height = np.linalg.norm(eye[2] - eye[4])
        height = (upper_height + lower_height) / 2
        
        # Eye area (using shoelace formula)
        area = 0.5 * abs(
            sum(eye[i][0]*eye[(i+1)%6][1] - eye[(i+1)%6][0]*eye[i][1] for i in range(6))
        )
        
        # Eye opening ratio (height/width)
        ratio = width / height if height != 0 else 0
        
        # Eye corner angles
        inner_angle = np.arctan2(eye[1][1] - eye[0][1], eye[1][0] - eye[0][0])
        outer_angle = np.arctan2(eye[2][1] - eye[3][1], eye[2][0] - eye[3][0])
        angle_diff = abs(inner_angle - outer_angle)
        
        return ratio, area, angle_diff
    
    # Calculate features for both eyes
    left_ratio, left_area, left_angle = eye_features(left_eye)
    right_ratio, right_area, right_angle = eye_features(right_eye)
    
    # Average measurements
    avg_ratio = (left_ratio + right_ratio) / 2
    avg_area = (left_area + right_area) / 2
    avg_angle = (left_angle + right_angle) / 2
    
    # Initialize scores dictionary
    scores = {}
    
    # Almond eyes criteria (Xie & Lam, 2015)
    # Almond eyes have specific ratio and angle characteristics
    almond_score = 0.85 - abs(avg_ratio - 2.4) * 1.2
    if 2.1 <= avg_ratio <= 2.7:  # Ideal ratio range
        almond_score += 0.15
    if avg_angle > 0.3:  # Upturned outer corners
        almond_score += 0.1
    scores['Almond'] = almond_score
    
    # Large eyes criteria (Ercan et al., 2008)
    # Large eyes have greater area and moderate ratio
    large_score = ((avg_area - 350) / 750) - 0.25
    if avg_area > 400:  # Significantly large area
        large_score += 0.2
    if 1.8 <= avg_ratio <= 2.2:  # Moderate ratio
        large_score += 0.1
    scores['Large'] = large_score
    
    # Small eyes criteria (Ercan et al., 2008)
    # Small eyes have smaller area and often rounder shape
    small_score = ((150 - avg_area) / 200) - 0.2
    if avg_area < 140:  # Significantly small area
        small_score += 0.2
    if avg_ratio < 1.8:  # Rounder shape
        small_score += 0.1
    scores['Small'] = small_score
    
    # Round eyes criteria (Xie & Lam, 2015)
    # Round eyes have more equal width and height
    if 1.7 <= avg_ratio <= 1.9:
        round_score = 0.8 - abs(avg_ratio - 1.8) * 2.5
        if avg_angle < 0.2:  # Less upturned corners
            round_score += 0.15
    else:
        round_score = -abs(avg_ratio - 1.8) * 2.5
    scores['Round'] = round_score
    
    # Find the best matching shape
    best_shape = max(scores, key=scores.get)
    best_score = scores[best_shape]
    
    # Check for borderline cases
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1 and abs(sorted_scores[0][1] - sorted_scores[1][1]) < 0.1:
        return f"{best_shape} (borderline with {sorted_scores[1][0]})"
    
    # Calculate confidence percentage
    confidence = min(max((best_score + 1) / 2, 0), 1) * 100
    return f"{best_shape} ({confidence:.0f}%)"

# main menu widget
class MainMenu(QWidget):
    def __init__(self, face_callback):
        super().__init__()
        self.setWindowTitle('Main Menu')
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        # add a title label
        title = QLabel('Face & Feature Shape Analyzer')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 22px; font-weight: bold; margin-bottom: 20px;')
        layout.addWidget(title)
        # add a vertical spacer
        layout.addSpacing(30)
        self.face_btn = QPushButton('Analyze Facial Features')
        self.face_btn.setFixedSize(220, 50)
        self.face_btn.setStyleSheet('font-size: 16px;')
        self.exit_btn = QPushButton('Exit')
        self.exit_btn.setFixedSize(220, 40)
        self.exit_btn.setStyleSheet('font-size: 15px;')
        # center the buttons in a vertical layout
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(self.face_btn)
        btn_layout.addSpacing(15)
        btn_layout.addWidget(self.exit_btn)
        layout.addLayout(btn_layout)
        layout.addStretch()
        self.setLayout(layout)
        self.face_btn.clicked.connect(face_callback)
        self.exit_btn.clicked.connect(self.exit_all)
        self.resize(400, 320)

    # exit the application
    def exit_all(self):
        QApplication.quit()

# main face analyzer widget
class FaceAnalyzerApp(QWidget):
    def __init__(self, back_callback=None):
        super().__init__()
        self.setWindowTitle('Face, Feature & Emotion Analyzer')
        # face shape title label (bold, above webcam)
        self.face_shape_title = QLabel('Face & Feature Shape and Emotion Analysis')
        self.face_shape_title.setAlignment(Qt.AlignCenter)
        self.face_shape_title.setStyleSheet('font-size: 24px; font-weight: bold; margin-bottom: 10px;')
        self.image_label = QLabel()
        self.result_label = QLabel('Results will appear here.')
        # checkboxes for toggling overlays
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
        self.cb_emotion = QCheckBox('Show Emotion Label')
        self.cb_emotion.setChecked(False)
        toggles_layout = QHBoxLayout()
        toggles_layout.addWidget(self.cb_face_shape)
        toggles_layout.addWidget(self.cb_lip_shape)
        toggles_layout.addWidget(self.cb_nose_shape)
        toggles_layout.addWidget(self.cb_eye_shape)
        toggles_layout.addWidget(self.cb_landmarks)
        toggles_layout.addWidget(self.cb_emotion)
        layout = QVBoxLayout()
        # add face shape title above webcam feed
        layout.addWidget(self.face_shape_title)
        # center the webcam feed
        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        image_layout.addStretch()
        layout.addLayout(image_layout)
        layout.addWidget(self.result_label)
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
        self.cb_emotion.stateChanged.connect(self.update_frame)
        self.start_webcam()

    # start webcam capture
    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.timer.isActive():
            self.timer.start(30)

    # update frame from webcam and process overlays
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        orig_frame = frame.copy()
        gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        result_text = "No face detected."
        emotion_text = ""
        face_shape_label = "(undetected)"
        for face in faces:
            shape = landmark_predictor(gray, face)
            landmarks = shape_to_np(shape)
            face_shape_label = analyze_face_shape(landmarks)
            if self.cb_face_shape.isChecked():
                face_outline = np.array(landmarks[0:17], np.int32)
                cv2.polylines(frame, [face_outline], False, (0,0,255), 2)
                cv2.putText(frame, f"Face: {face_shape_label}", (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
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
            if self.cb_emotion.isChecked():
                # emotion detection on face roi
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                # ensure roi is within image bounds
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + w, gray.shape[1]), min(y + h, gray.shape[0])
                face_roi = gray[y1:y2, x1:x2]
                try:
                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        face_roi = cv2.resize(face_roi, (64, 64))
                        face_roi = face_roi.astype("float32") / 255.0
                        if len(face_roi.shape) == 2:
                            face_roi = np.expand_dims(face_roi, axis=-1)  # (64, 64, 1)
                        face_roi = np.expand_dims(face_roi, axis=0)  # (1, 64, 64, 1)
                        preds = emotion_classifier.predict(face_roi, verbose=0)[0]
                        emotion_probability = np.max(preds)
                        emotion_label = emotion_labels[preds.argmax()]
                        emotion_text = f"Emotion: {emotion_label} ({emotion_probability*100:.1f}%)"
                        label_pos = (x, max(y - 20, 20))
                        cv2.putText(frame, emotion_text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
                        cv2.putText(frame, emotion_text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                        # overlay emotion-specific emoji png beside label
                        emoji_img = emoji_imgs.get(emotion_label)
                        if emoji_img is not None:
                            emoji_h = 48  # recommended: 44-56px for visibility
                            emoji_w = int(emoji_img.shape[1] * (emoji_h / emoji_img.shape[0]))
                            emoji_resized = cv2.resize(emoji_img, (emoji_w, emoji_h), interpolation=cv2.INTER_AREA)
                            ex = label_pos[0] - emoji_w - 10  # left of label, with 10px gap
                            ey = label_pos[1] - int(emoji_h/2)
                            if ex >= 0 and ey+emoji_h < frame.shape[0] and ey >= 0:
                                overlay = frame[ey:ey+emoji_h, ex:ex+emoji_w]
                                alpha_emoji = emoji_resized[:,:,3] / 255.0
                                alpha_bg = 1.0 - alpha_emoji
                                for c in range(3):
                                    overlay[:,:,c] = (alpha_emoji * emoji_resized[:,:,c] + alpha_bg * overlay[:,:,c])
                                frame[ey:ey+emoji_h, ex:ex+emoji_w] = overlay
                    else:
                        emotion_text = "Emotion: (undetected)"
                except Exception:
                    emotion_text = "Emotion: (undetected)"
            result_text = f"Face: {face_shape_label}\nLips: {analyze_lip_shape(landmarks)}\nNose: {analyze_nose_shape(landmarks)}\nEyes: {analyze_eye_shape(landmarks)}"
            break
        if self.cb_emotion.isChecked():
            result_text += f"\n{emotion_text if emotion_text else 'Emotion: (undetected)'}"
        self.face_shape_title.setText(f"Face Shape: {face_shape_label}")
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        self.result_label.setText(result_text)

    # release webcam and close windows
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        cv2.destroyAllWindows()
        event.accept()

# main controller for switching between menu and analyzer
class MainController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.menu = MainMenu(self.show_face)
        self.face_window = None
        self.menu.show()

    # show face analyzer window
    def show_face(self):
        if self.face_window is None:
            self.face_window = FaceAnalyzerApp(self.show_menu)
        self.menu.hide()
        self.face_window.show()

    # show main menu window
    def show_menu(self):
        if self.face_window:
            self.face_window.hide()
        self.menu.show()

    # run the application event loop
    def run(self):
        sys.exit(self.app.exec_())

if __name__ == '__main__':
    try:
        controller = MainController()
        controller.run()
    except SystemExit:
        try:
            if hasattr(controller, 'face_window') and controller.face_window and controller.face_window.cap:
                controller.face_window.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass