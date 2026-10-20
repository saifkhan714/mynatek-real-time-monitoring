import sys
import cv2
import numpy as np
import json
import logging
from datetime import datetime
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QMessageBox, QDesktopWidget
from pynput.mouse import Listener as MouseListener, Button
import os
from pathlib import Path
import argparse

# Global variables to track mouse clicks and locations
mouse_clicked = False
mouse_xy = (0, 0)
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
        if label_type is not None:
            cv2.putText(frame, f'Label: {label_type}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def cleanup(self):
        self.cap.release()


class VideoHandlerApp(QMainWindow):
    def __init__(self, video_path, verbose=False, webcam=None, start_frame=0):
        super().__init__()

        self.video_handler = VideoHandler(video_path, verbose, webcam, start_frame)

        self.setWindowTitle("Video Labeling Tool")
        self.resize(900, 700)

        # Video display area
        self.video_frame = QLabel(self)
        self.video_frame.setAlignment(Qt.AlignCenter)

        # Control buttons in a horizontal layout
        controls_widget = QWidget(self)
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setSpacing(8)

        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setStyleSheet("background-color: black; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.play_button)

        self.speed_buttons = []
        self.speeds = [0.25, 0.5, 0.75, 1.0, 1.25]
        for speed in self.speeds:
            btn = QPushButton(f"{speed:.2f}x", self)
            btn.setStyleSheet("background-color: #333; color: white; font-weight: bold; border-radius: 5px;")
            btn.clicked.connect(lambda checked, s=speed: self.set_playback_speed(s))
            self.speed_buttons.append(btn)
            controls_layout.addWidget(btn)

        self.back5_button = QPushButton("<<5", self)
        self.back5_button.clicked.connect(self.step_back_5s)
        self.back5_button.setStyleSheet("background-color: #333; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.back5_button)

        self.forward5_button = QPushButton(">>5", self)
        self.forward5_button.clicked.connect(self.step_forward_5s)
        self.forward5_button.setStyleSheet("background-color: #333; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.forward5_button)

        self.frame_back_button = QPushButton("Frame Back", self)
        self.frame_back_button.clicked.connect(self.step_back_frame)
        self.frame_back_button.setStyleSheet("background-color: #333; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.frame_back_button)

        self.frame_forward_button = QPushButton("Frame Forward", self)
        self.frame_forward_button.clicked.connect(self.step_forward_frame)
        self.frame_forward_button.setStyleSheet("background-color: #333; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.frame_forward_button)

        self.speed_label = QLabel("Speed: 1.00x", self)
        self.speed_label.setStyleSheet("color: white; font-weight: bold; background: transparent;")
        controls_layout.addWidget(self.speed_label)

        self.save_button = QPushButton("Save Annotations", self)
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setStyleSheet("background-color: black; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.save_button)

        # Restart Video button
        self.restart_button = QPushButton("Restart", self)
        self.restart_button.clicked.connect(self.restart_video)
        self.restart_button.setStyleSheet("background-color: #444; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.restart_button)

        # Full Screen button
        self.fullscreen_button = QPushButton("Full Screen", self)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setStyleSheet("background-color: #222; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.fullscreen_button)

        self.cleaning_button = QPushButton("Label as Cleaning", self)
        self.cleaning_button.clicked.connect(self.label_as_cleaning)
        self.cleaning_button.setStyleSheet("background-color: black; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.cleaning_button)

        self.not_cleaning_button = QPushButton("Label as Not Cleaning", self)
        self.not_cleaning_button.clicked.connect(self.label_as_not_cleaning)
        self.not_cleaning_button.setStyleSheet("background-color: black; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.not_cleaning_button)

        self.playback_speed = 1.0

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_frame)
        main_layout.addWidget(controls_widget)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.mouse_data = []

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
                    "Click 'Play' to start the video.\n"
                    "Keyboard shortcuts:\n"
                    "  Space: Play/Pause\n"
                    "  n: Label as Not Cleaning\n"
                    "  c: Label as Cleaning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def keyPressEvent(self, event):
        was_fullscreen = self.isFullScreen()
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_N:
            self.label_as_not_cleaning()
        elif event.key() == Qt.Key_C:
            self.label_as_cleaning()
        elif event.key() == Qt.Key_1:
            self.set_playback_speed(0.125)
        elif event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)
        # Restore full screen if it was active before the key event
        if was_fullscreen and not self.isFullScreen():
            self.showFullScreen()
            self.fullscreen_button.setText("Exit Full Screen")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("Full Screen")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("Exit Full Screen")

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(int(1000 / (self.video_handler.reported_fps * self.playback_speed)))
            self.play_button.setText("Pause")

    def restart_video(self):
        self.timer.stop()
        self.play_button.setText("Play")
        self.video_handler.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_handler.frame_num = 0
        frame = self.video_handler.read_frame()
        if frame is not None:
            annotated_frame = self.video_handler.annotate_frame_text()
            h, w, c = annotated_frame.shape
            bytes_per_line = c * w
            q_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            # Get available screen size
            screen = QDesktopWidget().screenGeometry()
            max_width = screen.width()
            max_height = screen.height()
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                min(w, max_width),
                min(h, max_height),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_frame.setPixmap(scaled_pixmap)
        # Do NOT clear mouse_data; all labeling changes are preserved

    def set_playback_speed(self, speed):
        self.playback_speed = speed
        self.speed_label.setText(f"Speed: {self.playback_speed:.2f}x")
        if self.timer.isActive():
            self.timer.start(int(1000 / (self.video_handler.reported_fps * self.playback_speed)))

    def step_back_5s(self):
        # Go back 5 seconds
        fps = self.video_handler.reported_fps
        current_pos = self.video_handler.cap.get(cv2.CAP_PROP_POS_FRAMES)
        frames_to_jump = int(fps * 5)
        new_pos = max(0, current_pos - frames_to_jump)

        # Get the last known label before the jump
        last_known_label = label_type if label_type is not None else "Not-Cleaning"

        # Update all frames in between with the last known label
        for frame_num in range(int(new_pos), int(current_pos)):
            entry = {
                'frame_num': frame_num,
                'clicked': False,
                'x': 0,
                'y': 0,
                'label': last_known_label
            }
            # Update or append entry for this frame_num
            for i, d in enumerate(self.mouse_data):
                if d['frame_num'] == frame_num:
                    self.mouse_data[i] = entry
                    break
            else:
                self.mouse_data.append(entry)

        self.video_handler.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        self.video_handler.frame_num = int(new_pos)
        frame = self.video_handler.read_frame()
        if frame is not None:
            annotated_frame = self.video_handler.annotate_frame_text()
            h, w, c = annotated_frame.shape
            bytes_per_line = c * w
            q_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.video_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_frame.setPixmap(scaled_pixmap)

    def step_forward_5s(self):
        # Go forward 5 seconds
        fps = self.video_handler.reported_fps
        current_pos = self.video_handler.cap.get(cv2.CAP_PROP_POS_FRAMES)
        frames_to_jump = int(fps * 5)
        new_pos = min(self.video_handler.length - 1, current_pos + frames_to_jump)

        # Get the last known label before the jump
        last_known_label = label_type if label_type is not None else "Not-Cleaning"

        # Update all frames in between with the last known label
        for frame_num in range(int(current_pos), int(new_pos)):
            entry = {
                'frame_num': frame_num,
                'clicked': False,
                'x': 0,
                'y': 0,
                'label': last_known_label
            }
            # Update or append entry for this frame_num
            for i, d in enumerate(self.mouse_data):
                if d['frame_num'] == frame_num:
                    self.mouse_data[i] = entry
                    break
            else:
                self.mouse_data.append(entry)

        self.video_handler.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        self.video_handler.frame_num = int(new_pos)
        frame = self.video_handler.read_frame()
        if frame is not None:
            annotated_frame = self.video_handler.annotate_frame_text()
            h, w, c = annotated_frame.shape
            bytes_per_line = c * w
            q_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.video_frame.setPixmap(QPixmap.fromImage(q_image))

    def step_back_frame(self):
        # Go back one frame
        current_pos = self.video_handler.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = max(0, current_pos - 2)  # -2 because read_frame() will increment by 1

        # Get the last known label before the jump
        last_known_label = label_type if label_type is not None else "Not-Cleaning"

        # Update the frame with the last known label
        entry = {
            'frame_num': int(new_pos),
            'clicked': False,
            'x': 0,
            'y': 0,
            'label': last_known_label
        }
        # Update or append entry for this frame_num
        for i, d in enumerate(self.mouse_data):
            if d['frame_num'] == int(new_pos):
                self.mouse_data[i] = entry
                break
        else:
            self.mouse_data.append(entry)

        self.video_handler.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        self.video_handler.frame_num = int(new_pos)
        frame = self.video_handler.read_frame()
        if frame is not None:
            annotated_frame = self.video_handler.annotate_frame_text()
            h, w, c = annotated_frame.shape
            bytes_per_line = c * w
            q_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.video_frame.setPixmap(QPixmap.fromImage(q_image))

    def step_forward_frame(self):
        # Go forward one frame
        current_pos = self.video_handler.cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Get the last known label before the jump
        last_known_label = label_type if label_type is not None else "Not-Cleaning"

        # Update the frame with the last known label
        entry = {
            'frame_num': int(current_pos),
            'clicked': False,
            'x': 0,
            'y': 0,
            'label': last_known_label
        }
        # Update or append entry for this frame_num
        for i, d in enumerate(self.mouse_data):
            if d['frame_num'] == int(current_pos):
                self.mouse_data[i] = entry
                break
        else:
            self.mouse_data.append(entry)

        frame = self.video_handler.read_frame()
        if frame is not None:
            annotated_frame = self.video_handler.annotate_frame_text()
            h, w, c = annotated_frame.shape
            bytes_per_line = c * w
            q_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.video_frame.setPixmap(QPixmap.fromImage(q_image))

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

        # Get available screen size
        screen = QDesktopWidget().screenGeometry()
        max_width = screen.width()
        max_height = screen.height()

        # Scale video to fit within screen, preserving aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            min(w, max_width),
            min(h, max_height),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_frame.setPixmap(scaled_pixmap)

        # Handle mouse click data and relabeling
        frame_num = self.video_handler.frame_num
        entry = {
            'frame_num': frame_num,
            'clicked': mouse_clicked,
            'x': mouse_xy[0] if mouse_clicked else 0,
            'y': mouse_xy[1] if mouse_clicked else 0,
            'label': label_type if label_type is not None else "Not-Cleaning"
        }
        # Overwrite or append entry for this frame_num
        for i, d in enumerate(self.mouse_data):
            if d['frame_num'] == frame_num:
                self.mouse_data[i] = entry
                break
        else:
            self.mouse_data.append(entry)
        mouse_clicked = False

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
