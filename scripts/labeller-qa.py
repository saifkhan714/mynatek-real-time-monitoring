import sys
import cv2
import json
from pathlib import Path
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QHBoxLayout, QDialog, QDialogButtonBox, QFormLayout, QComboBox,
    QDesktopWidget, QScrollArea
)
import argparse

class VideoLabelViewer(QMainWindow):
    def __init__(self, video_path, json_path):
        super().__init__()
        self.setWindowTitle("Video Label Viewer")

        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()

        # Set initial window size to 80% of screen
        self.resize(int(self.screen_width * 0.8), int(self.screen_height * 0.8))

        # Get screen dimensions
        screen = QDesktopWidget().screenGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()

        # Set initial window size to 80% of screen
        self.resize(int(self.screen_width * 0.8), int(self.screen_height * 0.8))

        self.video_path = video_path
        self.json_path = json_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        self.playing = False
        self.speed = 1.0

        # Load JSON labels
        with open(self.json_path, "r") as f:
            self.labels = json.load(f)

        # Map frame number to label
        self.label_map = {}
        for entry in self.labels:
            frame_num = entry.get("frame_num", 0)
            label = entry.get("label", entry.get("class", ""))
            self.label_map[frame_num] = label

        # Create scroll area for video frame
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Video display area
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(1, 1)
        self.scroll_area.setWidget(self.video_frame)

        # Info label with enhanced styling
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            background-color: rgba(45, 45, 45, 0.9);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        """)

        # Controls with enhanced styling
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        # Create buttons with enhanced styling
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        self.back_button = QPushButton("<< 5s", self)
        self.back_button.clicked.connect(self.jump_back)
        controls_layout.addWidget(self.back_button)

        self.forward_button = QPushButton("5s >>", self)
        self.forward_button.clicked.connect(self.jump_forward)
        controls_layout.addWidget(self.forward_button)

        # Speed control buttons with enhanced styling
        self.speed_1 = QPushButton("1x", self)
        self.speed_1.clicked.connect(lambda: self.set_speed(1.0))
        self.speed_1.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
                border-color: #5d5d5d;
            }
        """)
        controls_layout.addWidget(self.speed_1)

        self.speed_05 = QPushButton("0.5x", self)
        self.speed_05.clicked.connect(lambda: self.set_speed(0.5))
        self.speed_05.setStyleSheet(self.speed_1.styleSheet())
        controls_layout.addWidget(self.speed_05)

        self.speed_025 = QPushButton("0.25x", self)
        self.speed_025.clicked.connect(lambda: self.set_speed(0.25))
        self.speed_025.setStyleSheet(self.speed_1.styleSheet())
        controls_layout.addWidget(self.speed_025)

        self.speed_0125 = QPushButton("0.125x", self)
        self.speed_0125.clicked.connect(lambda: self.set_speed(0.125))
        self.speed_0125.setStyleSheet(self.speed_1.styleSheet())
        controls_layout.addWidget(self.speed_0125)

        self.change_label_button = QPushButton("Change Label", self)
        self.change_label_button.clicked.connect(self.change_label_dialog)
        controls_layout.addWidget(self.change_label_button)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_labels)
        controls_layout.addWidget(self.save_button)

        self.fullscreen_button = QPushButton("Full Screen", self)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        controls_layout.addWidget(self.fullscreen_button)

        self.fullscreen_button = QPushButton("Full Screen", self)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setStyleSheet("background-color: #222; color: white; font-weight: bold; border-radius: 5px;")
        controls_layout.addWidget(self.fullscreen_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.scroll_area)
        main_layout.addWidget(self.info_label)
        main_layout.addWidget(controls_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Timer for playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.update_frame()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_frame()  # Update frame scaling when window is resized

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("Full Screen")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("Exit Full Screen")
        self.update_frame()  # Update frame scaling after fullscreen toggle

    def toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
        else:
            # Ensure minimum speed is 0.125
            self.speed = max(0.125, self.speed)
            interval = int(1000 / (self.fps * self.speed))
            self.timer.start(interval)
            self.playing = True
            self.play_button.setText("Pause")

    def set_speed(self, speed):
        self.speed = speed
        print(f"[DEBUG] set_speed called: speed={self.speed}, playing={self.playing}")
        # Ensure minimum speed is 0.125
        self.speed = max(0.125, speed)
        if self.playing:
            self.timer.stop()
            interval = int(1000 / (self.fps * self.speed))
            self.timer.start(interval)

    def jump_back(self):
        target_frame = max(self.current_frame - int(self.fps * 5), 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.current_frame = target_frame
        self.update_frame()

    def jump_forward(self):
        target_frame = min(self.current_frame + int(self.fps * 5), self.frame_count - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.current_frame = target_frame
        self.update_frame()

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")
            return
        self.current_frame += 1
        self.display_frame(frame)

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)

    def display_frame(self, frame):
        # Overlay info
        timestamp_sec = self.current_frame / self.fps
        minutes, seconds = divmod(timestamp_sec, 60)
        hours, minutes = divmod(minutes, 60)
        timestamp_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        # Get label for current frame without modifying it
        label = self.label_map.get(self.current_frame, "")

        overlay = frame.copy()
        cv2.putText(overlay, f"Frame: {self.current_frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(overlay, f"Timestamp: {timestamp_str}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0), 2)
        cv2.putText(overlay, f"Label: {label}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Convert to QImage
        rgb_image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Calculate scaled size while maintaining aspect ratio
        available_width = self.scroll_area.viewport().width()
        available_height = self.scroll_area.viewport().height()

        # Calculate scaling factors
        width_ratio = available_width / w
        height_ratio = available_height / h

        # Use the smaller ratio to ensure the image fits
        scale_ratio = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(w * scale_ratio)
        new_height = int(h * scale_ratio)

        # Create scaled pixmap
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Set the pixmap
        self.video_frame.setPixmap(scaled_pixmap)

        # Update info label
        self.info_label.setText(f"Frame: {self.current_frame} / {self.frame_count-1} | Timestamp: {timestamp_str} | Label: {label}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_A:
            # Jump back 5 seconds
            target_frame = max(self.current_frame - int(self.fps * 5), 0)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame = target_frame
            self.update_frame()
        elif event.key() == Qt.Key_D:
            # Jump forward 5 seconds
            target_frame = min(self.current_frame + int(self.fps * 5), self.frame_count - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame = target_frame
            self.update_frame()
        elif event.key() == Qt.Key_C:
            # Mark as Cleaning
            frame_num = self.current_frame
            self.label_map[frame_num] = "Cleaning"
            # Update in self.labels
            if frame_num < len(self.labels):
                if "label" in self.labels[frame_num]:
                    self.labels[frame_num]["label"] = "Cleaning"
                elif "class" in self.labels[frame_num]:
                    self.labels[frame_num]["class"] = "Cleaning"
                else:
                    self.labels[frame_num]["label"] = "Cleaning"
            self.update_frame()
        elif event.key() == Qt.Key_N:
            # Mark as Not-Cleaning
            frame_num = self.current_frame
            self.label_map[frame_num] = "Not-Cleaning"
            # Update in self.labels
            if frame_num < len(self.labels):
                if "label" in self.labels[frame_num]:
                    self.labels[frame_num]["label"] = "Not-Cleaning"
                elif "class" in self.labels[frame_num]:
                    self.labels[frame_num]["class"] = "Not-Cleaning"
                else:
                    self.labels[frame_num]["label"] = "Not-Cleaning"
            self.update_frame()
        elif event.key() == Qt.Key_1:
            self.set_speed(1.0)
        elif event.key() == Qt.Key_2:
            self.set_speed(0.75)
        elif event.key() == Qt.Key_3:
            self.set_speed(0.5)
        elif event.key() == Qt.Key_4:
            self.set_speed(0.25)
        elif event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def change_label_dialog(self):
        # Pause the video if playing
        if self.playing:
            self.toggle_play()

        # Use current frame number directly
        frame_num = self.current_frame
        old_label = self.label_map.get(frame_num, "")
        dlg = QDialog(self)
        dlg.setWindowTitle("Change Label")
        dlg.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 5px;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #4d4d4d;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #4d4d4d;
                border-color: #5d5d5d;
            }
        """)

        form_layout = QFormLayout(dlg)
        form_layout.setSpacing(15)
        form_layout.setContentsMargins(20, 20, 20, 20)

        # Bold formatting for frame number, old label, and new label
        frame_label = QLabel(f"<b>Frame:</b> <b>{frame_num}</b>")
        old_label_label = QLabel(f"<b>Current Label:</b> <b>{old_label}</b>")
        new_label_label = QLabel(f"<b>New Label:</b> ")
        new_label_combo = QComboBox()
        new_label_combo.addItems(["Cleaning", "Not-Cleaning"])
        # Set current index to match old label if possible
        if old_label == "Cleaning":
            new_label_combo.setCurrentIndex(0)
        elif old_label == "Not-Cleaning":
            new_label_combo.setCurrentIndex(1)
        form_layout.addRow(frame_label)
        form_layout.addRow(old_label_label)
        form_layout.addRow(new_label_label, new_label_combo)

        # OK/Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        form_layout.addWidget(buttons)

        def on_accept():
            new_label = new_label_combo.currentText()
            # Only update if label has changed
            if new_label != old_label:
                # Update in-memory label map and self.labels
                self.label_map[frame_num] = new_label
                # Also update in self.labels (assume list of dicts)
                if frame_num < len(self.labels):
                    if "label" in self.labels[frame_num]:
                        self.labels[frame_num]["label"] = new_label
                    elif "class" in self.labels[frame_num]:
                        self.labels[frame_num]["class"] = new_label
                    else:
                        self.labels[frame_num]["label"] = new_label
            self.update_frame()
            dlg.accept()

        def on_reject():
            dlg.reject()

        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(on_reject)

        dlg.exec_()

    def save_labels(self):
        # Create a new list of labels with exact frame numbers
        updated_labels = []
        for frame_num in range(self.frame_count):
            label = self.label_map.get(frame_num, "")
            entry = {
                "frame_num": frame_num,
                "label": label
            }
            updated_labels.append(entry)

        # Save to JSON file
        with open(self.json_path, "w") as f:
            json.dump(updated_labels, f, indent=2)
        print(f"Labels saved to {self.json_path}")

    def closeEvent(self, event):
        self.cap.release()
        QApplication.quit()
        event.accept()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Show video with labels from JSON")
    parser.add_argument('--video-path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--json-path', type=str, required=True, help='Path to the JSON file with labels')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = VideoLabelViewer(args.video_path, args.json_path)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
