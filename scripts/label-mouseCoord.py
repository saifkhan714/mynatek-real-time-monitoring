import sys
import cv2
import json
import logging
from datetime import datetime
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QMessageBox
from pynput.mouse import Listener as MouseListener, Button
import os
from pathlib import Path
import argparse

# Global variables to track mouse clicks and locations
mouse_clicked = False
mouse_xy = (0, 0)
label_type = None  # "Cleaning" or "Not Cleaning"
sequence_data = []  # Store data for multiple sequences

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

        # Initialize start_video_time here
        self.start_video_time = datetime.now()  # Initialize current time

    def read_frame(self, skip_frames=True):
        ret, self.frame = self.cap.read()
        if not ret:
            return None

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

        # Ensure mouse_xy is an integer tuple before using it in the cv2.circle function
        mouse_x, mouse_y = int(mouse_xy[0]), int(mouse_xy[1])  # Cast to integers

        # Display the mouse click location on the frame
        if mouse_xy != (0, 0):  # Only draw the circle if the mouse was clicked
            cv2.circle(frame, (mouse_x, mouse_y), 10, (0, 0, 255), -1)  # Red circle for mouse click

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

        self.mouse_data = []
        self.sequence_id = 1  # Keep track of current sequence

        # Timer to update frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Initially stop the timer, so video doesn't play
        self.timer.stop()  # Ensure the timer doesn't start on app launch

        # Show a popup message about the labeling tool
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
            self.timer.start(int(1000 / self.video_handler.reported_fps))  # Corrected the float to int
            self.play_button.setText("Pause")

    def update_frame(self):
        global mouse_clicked, mouse_xy, label_type  # Declare the global variables here
        
        frame = self.video_handler.read_frame()
        if frame is None:
            self.timer.stop()
            return

        # Annotate the frame with text
        annotated_frame = self.video_handler.annotate_frame_text()

        # Convert to QImage and display in PyQt widget
        h, w, c = annotated_frame.shape
        bytes_per_line = c * w
        q_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_frame.setPixmap(QPixmap.fromImage(q_image))

        # Handle mouse click data
        if mouse_clicked:
            if mouse_xy != (0, 0):  # Only save non-zero positions
                self.mouse_data.append({
                    'sequence_id': f"action_{self.sequence_id}",
                    'frame_num': self.video_handler.frame_num,
                    'keypoints': [{'x': mouse_xy[0], 'y': mouse_xy[1], 'confidence': 1.0}],  # confidence is 1 on valid click
                    'label': label_type
                })
            mouse_clicked = False  # Reset after saving the click

        else:
            # When no mouse click, save zero values for keypoints
            self.mouse_data.append({
                'sequence_id': f"action_{self.sequence_id}",
                'frame_num': self.video_handler.frame_num,
                'keypoints': [{'x': mouse_xy[0], 'y': mouse_xy[1], 'confidence': 0}],  # save current mouse position even if not clicked
                'label': label_type
            })

    def save_data(self):
        # Save mouse data as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fn = f'{Path(self.video_handler.video_path).stem}_annotations_{timestamp}.json'

        # Write data to JSON file
        with open(fn, 'w') as file:
            json.dump(self.mouse_data, file, indent=4)

        logging.info(f"Saved mouse data to {fn}, length of mouse_data: {len(self.mouse_data)}")


    def closeEvent(self, event):
        self.video_handler.cleanup()

    def label_as_cleaning(self):
        global label_type
        label_type = "Cleaning"

    def label_as_not_cleaning(self):
        global label_type
        label_type = "Not-Cleaning"


# Mouse Listener for clicks
def on_click(x, y, button, pressed):
    global mouse_clicked, mouse_xy
    if button == Button.left:
        mouse_clicked = pressed
        mouse_xy = (x, y)


# Start the mouse listener
mouse_listener = MouseListener(on_click=on_click)
mouse_listener.start()


def main():
    parser = argparse.ArgumentParser(description="Video Labeling Tool")
    parser.add_argument('--video-path', type=str, required=True, help='Path to the video file to label')
    args = parser.parse_args()
    app = QApplication(sys.argv)
    video_path = args.video_path
    ex = VideoHandlerApp(video_path)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
