#!/usr/bin/env python3
# pose_segment_label_gui.py
# --------------------------------------------------------------------
# GUI for labelling temporal segments as "cleaning" / "not_cleaning".
# It does *not* store key-points; just start/end frame indices per label.
# --------------------------------------------------------------------
import sys, json, uuid, logging, argparse
from datetime import datetime
from pathlib   import Path

import cv2
from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QMessageBox, QFileDialog)

# -------------------------------------------------------------------- helpers
class VideoHandler:
    """Light wrapper around cv2.VideoCapture."""
    def __init__(self, path_or_cam):
        self.cap = cv2.VideoCapture(path_or_cam)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {path_or_cam}")
        self.fps  = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.idx  = -1          # current frame number (0-based)
        self.path = str(path_or_cam)

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.idx += 1
        return ok, frame

    def release(self):
        self.cap.release()

# -------------------------------------------------------------------- main GUI
class SegmentLabelGUI(QMainWindow):

    def __init__(self, video_path: str):
        super().__init__()
        self.setWindowTitle("Cleaning / Not-Cleaning – Segment Annotator")
        self.setGeometry(80, 60, 1280, 720)

        # data
        self.video        = VideoHandler(video_path)
        self.active_seq   = None   # open interval or None
        self.sequences    = []     # finished intervals

        # ---------------------------------------------------------------- UI
        self.frame_label = QLabel(alignment=Qt.AlignCenter)
        self.setCentralWidget(self.frame_label)

        row_y = 650
        self.btn_play      = self._mk_btn("Play",            30,  row_y, self.toggle_play)
        self.btn_cleaning  = self._mk_btn("Cleaning",       150,  row_y, lambda: self.start_sequence("cleaning"))
        self.btn_notclean  = self._mk_btn("Not Cleaning",   280,  row_y, lambda: self.start_sequence("not_cleaning"))
        self.btn_endseq    = self._mk_btn("End Sequence",   450,  row_y, self.end_sequence)
        self.btn_save      = self._mk_btn("Save",           580,  row_y, self.save)

        # timer
        self.timer = QTimer(interval=int(1000 / self.video.fps))
        self.timer.timeout.connect(self.next_frame)

        self._style_buttons(default="not_cleaning")
        self._show_info_popup()
        self.show()

    # ----------------------------------------------------------- button helpers
    def _mk_btn(self, text, x, y, slot):
        b = QPushButton(text, self)
        b.setGeometry(x, y, 110, 34)
        b.clicked.connect(slot)
        b.setStyleSheet("background:#111;color:#fff;font-weight:bold;border-radius:4px;")
        return b

    def _style_buttons(self, default:str):
        # highlight the active label button
        active = self.btn_cleaning if default=="cleaning" else self.btn_notclean
        for b in (self.btn_cleaning, self.btn_notclean):
            b.setStyleSheet("background:#111;color:#fff;font-weight:bold;border-radius:4px;")
        active.setStyleSheet("background:#007acc;color:#fff;font-weight:bold;border-radius:4px;")

    def _show_info_popup(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("How to use")
        msg.setText(
            "▶  Play / Pause the video\n"
            "🟢  Click 'Cleaning' or 'Not Cleaning' to START a segment.\n"
            "🔴  Click 'End Sequence' to FINISH the current segment.\n"
            "💾  Click 'Save' to write a *_segments_*.json file."
        )
        msg.exec_()

    # ----------------------------------------------------------- controls
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start()
            self.btn_play.setText("Pause")

    def start_sequence(self, label:str):
        # close previous if still open
        if self.active_seq is not None:
            self.end_sequence()

        self.active_seq = {
            "sequence_id": f"seq_{uuid.uuid4().hex[:6]}",
            "label": label,
            "start_frame": self.video.idx + 1  # next frame to be read
        }
        # highlight
        self._style_buttons(default=label)

    def end_sequence(self):
        if self.active_seq is None:
            return
        # current frame has already been drawn; use idx as end
        self.active_seq["end_frame"] = self.video.idx
        self.sequences.append(self.active_seq)
        self.active_seq = None
        # reset button highlight
        self._style_buttons(default="not_cleaning")

    def save(self):
        # auto-close any running sequence
        if self.active_seq:
            self.end_sequence()

        if not self.sequences:
            QMessageBox.warning(self, "Nothing to save", "No segments recorded yet.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = f"{Path(self.video.path).stem}_segments_{ts}.json"
        payload = {
            "segments": [
                {
                    "video_id": Path(self.video.path).stem,
                    **seq
                } for seq in self.sequences
            ],
            "meta": {
                "fps": int(self.video.fps)
            }
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        QMessageBox.information(self, "Saved",
                                f"Saved {len(self.sequences)} segments → {out_json}")

    # ----------------------------------------------------------- main loop
    def next_frame(self):
        ok, frame = self.video.read()
        if not ok:
            self.timer.stop()
            return

        # draw current label (if any) on the frame
        lbl = self.active_seq["label"] if self.active_seq else "idle"
        cv2.putText(frame, lbl.upper(), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # show frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(qimg))

    # ----------------------------------------------------------- cleanup
    def closeEvent(self, evt):
        self.video.release()
        evt.accept()

# -------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Path to video file (leave empty to choose interactively)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    if not args.video:
        # pick a file
        fn, _ = QFileDialog.getOpenFileName(None, "Choose a video",
                                            "", "Videos (*.mp4 *.avi *.mov)")
        if not fn:
            sys.exit(0)
        args.video = fn

    gui = SegmentLabelGUI(args.video)
    sys.exit(app.exec_())

# --------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
