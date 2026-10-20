import sys
import cv2
import json
import logging
import argparse
from datetime import datetime
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QMessageBox
import os
from pathlib import Path
import copy
import numpy as np
from src import model
from src import util 
from src.body import Body

# Global variable for label
label_type = None  # "Cleaning" or "Not Cleaning"

class VideoHandler:
    def __init__(self, video_path, verbose=False, webcam=None, start_frame=0):
        self.video_path = video_path
        self.verbose = verbose
        self.webcam = webcam
        self.start_frame = start_frame
        self.frame_num = 0
        self.frame = None
        self.fps_list = []
        self.window_size = 10
        self.smoothed_fps = 0
        self.last_frame_read_time = 0
        self.current_skeleton = None  # Store skeleton data for current frame

        # Initialize body pose estimation
        self.body_estimation = Body('model/body_pose_model.pth')

        # Initialize video capture
        if self.webcam is None:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(self.webcam)

        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.reported_fps == 0:
            self.reported_fps = 30
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize start_video време
        self.start_video_time = datetime.now()

    def read_frame(self, skip_frames=True):
        ret, self.frame = self.cap.read()
        if not ret:
            return None

        # Apply body pose estimation
        candidate, subset = self.body_estimation(self.frame)
        canvas = copy.deepcopy(self.frame)
        self.frame = util.draw_bodypose(canvas, candidate, subset)

        # Extract skeleton data
        num_joints = 18  # Adjust based on your model
        if len(subset) > 0:
            person = subset[0]
            skeleton = []
            for joint in range(num_joints):
                if person[joint] != -1:
                    index = int(person[joint])
                    x, y, score = candidate[index][:3]
                    skeleton.append([x, y, score])
                else:
                    skeleton.append([0, 0, 0])
        else:
            skeleton = [[0, 0, 0] for _ in range(num_joints)]
        self.current_skeleton = skeleton

        self.frame_num += 1
        return self.frame

    def annotate_frame_text(self):
        current_time = datetime.now()
        time_delta = (current_time - self.start_video_time).total_seconds()
        minutes, seconds = divmod(time_delta, 60)
        time_delta_str = f"{int(minutes):02d}:{int(seconds):02d}"

        frame = self.frame.copy()
        cv2.putText(frame, f'Input: {self.video_path}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Time : {time_delta_str}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS  : {self.smoothed_fps:.1f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Frame: {self.frame_num}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if label_type is not None:
            cv2.putText(frame, f'Label: {label_type}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

class VideoHandlerApp(QMainWindow):
    def __init__(self, video_path, verbose=False, webcam=None, start_frame=0):
        super().__init__()

        self.video_handler = VideoHandler(video_path, verbose, webcam, start_frame)

        self.setWindowTitle("Video Labeling Tool")
        self.setGeometry(100, 100, 800, 600)

        # Video display area
        self.video_frame = QLabel(self)
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.video_frame)

        # Control buttons
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setGeometry(20, 500, 100, 30)
        self.play_button.setStyleSheet("background-color: black; color: white; font-weight: bold; border-radius: 5px;")

        self.save_button = QPushButton("Save Annotations", self)
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setGeometry(150, 500, 150, 30)
        self.save_button.setStyleSheet("background-color: black; color: white; font-weight: bold; border-radius: 5px;")

        self.cleaning_button = QPushButton("Label as Cleaning", self)
        self.cleaning_button.clicked.connect(self.label_as_cleaning)
        self.cleaning_button.setGeometry(320, 500, 150, 30)
        self.cleaning_button.setStyleSheet("background-color: black; color: white; font-weight: bold; border-radius: 5px;")

        self.not_cleaning_button = QPushButton("Label as Not Cleaning", self)
        self.not_cleaning_button.clicked.connect(self.label_as_not_cleaning)
        self.not_cleaning_button.setGeometry(490, 500, 150, 30)
        self.not_cleaning_button.setStyleSheet("background-color: black; color: white; font-weight: bold; border-radius: 5px;")

        self.frame_data = []  # Store frame data with skeletons and labels

        # Timer to update frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Initially stop the timer
        self.timer.stop()

        # Show info popup
        self.show_info_popup()

        self.show()

    def show_info_popup(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Welcome to the Labeling Tool")
        msg.setText("This tool allows you to label frames of a video as 'Cleaning' or 'Not Cleaning'.\n"
                    "Click 'Label as Cleaning' or 'Label as Not Cleaning' to start labeling.\n"
                    "Click 'Play' to start the video.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(int(1000 / self.video_handler.reported_fps))
            self.play_button.setText("Pause")

    def update_frame(self):
        frame = self.video_handler.read_frame()
        if frame is None:
            self.timer.stop()
            return

        # Save frame data if labeled
        skeleton = self.video_handler.current_skeleton
        if label_type is not None and skeleton is not None:
            self.frame_data.append({
                'frame_num': self.video_handler.frame_num,
                'skeleton': skeleton,
                'label': label_type
            })

        # Annotate and display frame
        annotated_frame = self.video_handler.annotate_frame_text()
        h, w, c = annotated_frame.shape
        bytes_per_line = c * w
        q_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_frame.setPixmap(QPixmap.fromImage(q_image))

    def save_data(self):
        sequences = []
        current_sequence = {'label': None, 'skeletons': []}
        for frame in self.frame_data:
            if frame['label'] != current_sequence['label']:
                if current_sequence['skeletons']:
                    sequences.append(current_sequence)
                current_sequence = {'label': frame['label'], 'skeletons': []}
            current_sequence['skeletons'].append(frame['skeleton'])
        if current_sequence['skeletons']:
            sequences.append(current_sequence)

        # Save to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fn = f'{Path(self.video_handler.video_path).stem}_sequences_{timestamp}.json'
        with open(fn, 'w') as file:
            json.dump(sequences, file, indent=4)
        logging.info(f"Saved sequences to {fn}, number of sequences: {len(sequences)}")

    def closeEvent(self, event):
        self.video_handler.cleanup()

    def label_as_cleaning(self):
        global label_type
        label_type = "Cleaning"

    def label_as_not_cleaning(self):
        global label_type
        label_type = "Not-Cleaning"

def main():
    parser = argparse.ArgumentParser(description='Video Labeling Tool')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the video file')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ex = VideoHandlerApp(args.video_path)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()