# Human Action Recognition Pipeline (YOLOv11 Pose + ByteTrack + ST-GCN)

This repository provides a complete, production-ready pipeline for human action recognition in videos using YOLOv11 pose estimation, ByteTrack tracking, and a custom-trained ST-GCN model.

---

## 1. Environment Setup

**This section explains how to set up your Python environment so that all dependencies are installed and compatible. This is critical for reproducibility and for running the code on both local machines and servers.**

### 1.1. Clone the Repository

Clone the repository to your local machine or server:
```sh
git clone <your-repo-url>
cd Pipeline_STGCN
```

### 1.2. Create and Activate Environment

We recommend using Conda for environment management. This ensures all dependencies (especially for PyTorch and PyTorch Geometric) are compatible.
```sh
conda create -n yolo_pose python=3.8 -y
conda activate yolo_pose
```

### 1.3. Install Requirements

Install all required Python packages:
```sh
pip install -r requirements.txt
```
**Note:** The requirements.txt is generated from a working environment and includes all necessary packages and versions.

#### PyTorch Geometric and Related Packages

These are required for ST-GCN and graph operations. Install them as follows:
```sh
pip install torch-geometric
conda install pytorch-spline-conv -c pyg
conda install pytorch-cluster -c pyg
conda install pytorch-scatter -c pyg
```
**Tip:** Always install these after the main requirements to avoid version conflicts.

---

## 2. Data Preparation

**This section covers how to prepare your data for training and inference. The pipeline expects pose/keypoint data in a specific format.**

### 2.1. Extract Pose Data

- Use YOLOv11 pose estimation to extract keypoints from your videos.
- Save the results as JSON files in `data/poses/`.
- Each JSON should be a list of frames, each with:
  - `"frame_num"`: frame index
  - `"label"`: action label (e.g., "Cleaning", "Not-Cleaning")
  - `"joints"`: list of 17 [x, y] pairs (keypoints for one person)
- **Tip:** Use the provided inference pipeline or your own script to generate these JSONs.

### 2.2. Convert JSON to ST-GCN PKL

Convert your JSON pose data to the format required for ST-GCN training:
```sh
python data/poses/json_to_stgcn_pkl.py
```
- This script will print progress and save `custom_data/cleaning_dataset.pkl`.
- It slides a window over frames to create sequences, stacks keypoints, and assigns labels.

### 2.3. Add Score Channel (OPTIONAL)

ST-GCN expects 3 channels (x, y, score). If your data only has (x, y), add a dummy score channel:
```sh
python custom_data/add_score_channel_to_pkl.py
```
- This script will print progress and save `custom_data/cleaning_dataset_with_score.pkl`.

---

## 3. Model Training

**This section explains how to train the ST-GCN model on your custom data.**

### 3.1. Train ST-GCN

```sh
python stgcn/train_custom.py --device cpu
# or for GPU:
python stgcn/train_custom.py --device cuda
```
- The script prints device info, data loading, and training progress.
- The trained model is saved to `stgcn_custom_saved/tsstg-model.pth`.
- You can adjust hyperparameters (epochs, batch size, learning rate) at the top of the script.

---

## 4. Model Testing

**This section explains how to evaluate your trained model on a test set.**

```sh
python stgcn/test_custom.py --device cpu
# or for GPU:
python stgcn/test_custom.py --device cuda
```
- The script prints device info, data loading, and evaluation progress.
- Outputs test accuracy, per-class accuracy, and a confusion matrix.

---

## 5. Inference (End-to-End on Video)

**This section explains how to run the full pipeline (YOLOv11 pose, ByteTrack, ST-GCN) on a video.**

```sh
python main.py --source short_test2.mp4 --device cpu --kpt-label --view-img --save-txt --save-conf --project runs/detect --name custom_infer
# or for GPU:
python main.py --source short_test2.mp4 --device cuda --kpt-label --view-img --save-txt --save-conf --project runs/detect --name custom_infer
```
- The script prints device info, model loading, video loading, and inference progress.
- Annotated video and results are saved in `runs/detect/custom_infer/`.
- All steps are clearly printed for user feedback.

---

## 6. Directory Structure

**This section shows how your project should be organized.**

```
project_root/
│
├── data/
│   ├── poses/
│   │   ├── webcam_feed_*.json
│   │   └── other JSON files
│   └── ...
├── data_preparation/
│   ├── json_to_stgcn_pkl.py
│   ├── add_score_channel_to_pkl.py
│   └── custom_data/
│       ├── cleaning_dataset_train.pkl
│       ├── cleaning_dataset_val.pkl
│       └── cleaning_dataset_test.pkl
│
├── stgcn/
│   ├── train_custom.py
│   ├── test_custom.py
│   ├── ActionsEstLoader.py
│   ├── Models.py
│   ├── Utils.py
│   └── pose_utils.py
│
├── stgcn_custom_saved/
│   └── tsstg-model.pth
│
├── bytetrack_utils/
│   ├── byte_tracker.py
│   ├── byte_tracker_new.py
│   ├── basetrack.py
│   ├── kalman_filter.py
│   ├── matching.py
│   └── visualize.py
│
├── utils/
│   ├── general.py
│   ├── metrics.py
│   ├── plots.py
│   └── torch_utils.py
│
├── weights/
│   └── yolo11m-pose.pt
│
├── runs/
│   └── detect/
│       └── custom_infer/
│           └── ...
│
├── main.py
├── requirements.txt
└── README.md
```
**Tip:** Always run scripts from the project root.

---

## 7. Key Scripts

**This section describes the purpose of each main script.**

- `data/poses/json_to_stgcn_pkl.py`: Converts JSON pose data to .pkl for ST-GCN. Prints progress and summary.
- `custom_data/add_score_channel_to_pkl.py`: Adds a dummy score channel to pose data. Prints before/after shapes and output path.
- `stgcn/train_custom.py`: Trains the ST-GCN model. Prints device, data loading, and training progress.
- `stgcn/test_custom.py`: Tests the ST-GCN model. Prints device, data loading, and evaluation progress.
- `main.py`: Runs YOLOv11 pose, ByteTrack, and ST-GCN for end-to-end inference. Prints device, model loading, video loading, and output saving.

---

## 8. Device Selection

**All scripts support both CPU and GPU. Use the `--device` argument to select:**
- `--device cpu` for CPU
- `--device cuda` for GPU

**Device info is printed at runtime for clarity.**

---

## 9. Troubleshooting

- Ensure all dependencies are installed as per `requirements.txt`.
- For PyTorch Geometric, use conda/pyg channels for best compatibility.
- Run all scripts from the project root directory.
- Check print statements for progress and device info at each step.
- If you encounter shape or device errors, check your data and script arguments.

---
