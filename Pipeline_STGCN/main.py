import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics import YOLO
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, check_requirements, check_imshow,
    non_max_suppression, scale_coords, xyxy2xywh,
    strip_optimizer, set_logging, increment_path, save_one_box
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from bytetrack_utils.byte_tracker_new import BYTETracker
import numpy as np
from stgcn.ActionsEstLoader import TSSTG


def detect(opt):
    # unpack options using parentheses to allow line breaks
    (source, view_img, save_txt, imgsz,
     save_txt_tidl, kpt_label, track_thresh,
     track_buffer, match_thresh) = (
        opt.source, opt.view_img, opt.save_txt,
        opt.img_size, opt.save_txt_tidl, opt.kpt_label,
        opt.track_thresh, opt.track_buffer, opt.match_thresh
    )
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')
    )

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    # Check for YOLOv11-m-pose model, download if not present
    yolo11m_pose_path = "weights/yolo11m-pose.pt"
    print("Preparing YOLOv11-m-pose model...")
    if not os.path.exists(yolo11m_pose_path):
        print(f"Model {yolo11m_pose_path} not found. Downloading with Ultralytics YOLO API...")
        model_dl = YOLO("yolo11m-pose.pt")  # This will auto-download to current dir or cache
        import shutil
        import glob
        # First, check if it was downloaded to the current directory
        if os.path.exists("yolo11m-pose.pt"):
            shutil.move("yolo11m-pose.pt", yolo11m_pose_path)
            print(f"Moved yolo11m-pose.pt to {yolo11m_pose_path}")
        # If not, search the cache
        elif not os.path.exists(yolo11m_pose_path):
            cache_dir = os.path.expanduser("~/.cache/ultralytics")
            found = False
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file == "yolo11m-pose.pt":
                        src = os.path.join(root, file)
                        shutil.copy(src, yolo11m_pose_path)
                        found = True
                        print(f"Copied {src} to {yolo11m_pose_path}")
                        break
                if found:
                    break
            if not found:
                raise FileNotFoundError("yolo11m-pose.pt was not found after download. Please check Ultralytics cache or current directory.")
    print("Loading YOLOv11-m-pose model...")
    model = YOLO(yolo11m_pose_path)
    stride = 32  # YOLOv11 default stride for pose models

    print("Initializing ByteTrack tracker...")
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh)
    frame_id = 0  # ct+++
    print(f"Loading ST-GCN action recognition model on device: {opt.device} ...")
    action_model = TSSTG(device=opt.device)

    if isinstance(imgsz, (list, tuple)):
        assert len(imgsz) == 2, "height and width of image must be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)

    names = model.names if hasattr(model, 'names') else None

    # Remove classify logic (not used with YOLOv11)

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        print(f"Loading webcam/stream source: {source}")
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        print(f"Loading video/image source: {source}")
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        # VideoWriter setup for saving output video
        first_path = None
        for item in dataset:
            first_path = item[0]
            first_im0 = item[2]
            break
        if first_path is not None and (first_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = first_im0.shape[:2]
            out_video_path = str(save_dir / (Path(first_path).stem + '_out.mp4'))
            vid_writer = cv2.VideoWriter(out_video_path, fourcc, 30, (w, h))
            print(f"Output video will be saved to: {out_video_path}")
        # Re-initialize dataset iterator after peeking first frame
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()

    print("Starting inference...")
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if frame_idx % 1 != 0:
            continue

        # Get the original image for Ultralytics YOLOv11 inference
        if webcam:
            im0 = im0s[0] if isinstance(im0s, list) else im0s
        else:
            im0 = im0s

        t1 = time_synchronized()
        results = model.predict(im0, conf=opt.conf_thres, iou=opt.iou_thres, verbose=False)
        t2 = time_synchronized()

        # Ultralytics returns a list of results, one per image
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
        # keypoints shape: (num_detections, 17, 2)

        Results = []
        Result_kpts = []
        for i in range(len(boxes)):
            xyxy = boxes[i]
            conf = scores[i]
            Results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)])
            # For tracking and ST-GCN, we need (17, 3): x, y, score (Ultralytics does not provide per-kpt score, so set to 1)
            kpts_xy = keypoints[i] if len(keypoints) > i else np.zeros((17, 2))
            kpts = np.concatenate([kpts_xy, np.ones((17, 1))], axis=1)  # (17, 3)
            Result_kpts.append(torch.tensor(kpts, dtype=torch.float32))

        online_tlwhs, online_ids, action_results = track_main(
            tracker, np.array(Results), Result_kpts,
            frame_id, 1080, 1920, (1080, 1920), action_model, im0
        )
        online_tlwhs = np.array(online_tlwhs).reshape(-1, 4)
        online_xyxys = tlwh2xyxy(online_tlwhs)
        outputs = np.concatenate((online_xyxys, np.array(online_ids).reshape(-1, 1)), axis=1) if len(online_ids) else []

        for det_index, (output, conf) in enumerate(zip(outputs, scores)):
            xyxy = output[:4]
            tid = output[4]
            if save_txt:
                bbox_left, bbox_top = xyxy[0], xyxy[1]
                bbox_w, bbox_h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                with open(f'{save_dir / "labels" / Path(path).stem}.txt', 'a') as f:
                    f.write(('%g ' * 10 + '\n') % (
                        frame_idx + 1, tid, bbox_left, bbox_top,
                        bbox_w, bbox_h, -1, -1, -1, det_index
                    ))
            if view_img:
                label = f'{int(tid)} {conf:.2f} {action_results[det_index] if det_index < len(action_results) else ""}'
                kpts = keypoints[det_index] if len(keypoints) > det_index else None
                # Ensure kpts is a 1D array of (17,3) for plot_one_box/plot_skeleton_kpts compatibility
                if kpts is not None:
                    if kpts.ndim == 1 and kpts.shape[0] == 34:
                        # (17,2) flattened, pad with conf=1.0
                        kpts = kpts.reshape(17, 2)
                        kpts = np.concatenate([kpts, np.ones((17, 1))], axis=1)
                        kpts = kpts.flatten()
                    elif kpts.ndim == 2 and kpts.shape[1] == 2:
                        # (17,2), pad with conf=1.0
                        kpts = np.concatenate([kpts, np.ones((17, 1))], axis=1)
                        kpts = kpts.flatten()
                    elif kpts.ndim > 1:
                        kpts = kpts.flatten()
                plot_one_box(
                    xyxy, im0, label=label, color=None,
                    line_thickness=opt.line_thickness,
                    kpt_label=kpt_label, kpts=kpts,
                    steps=3, orig_shape=im0.shape[:2]
                )

        print(f'Done. ({t2 - t1:.3f}s)')
        # Write frame to video if vid_writer is set
        if vid_writer is not None:
            vid_writer.write(im0)
        if view_img:
            cv2.imshow(str(path), im0)
            cv2.waitKey(1)

    # Release video writer if used (moved inside detect)
    if vid_writer is not None:
        vid_writer.release()
        print(f"Output video saved to: {out_video_path}")

    print("Inference complete.")

def track_main(tracker, detection_results, Result_kpts, frame_id,
               image_height, image_width, test_size,
               action_model, frame):
    online_targets = tracker.update(
        detection_results, Result_kpts,
        [image_height, image_width], test_size
    )
    online_tlwhs, online_ids, action_results = [], [], []

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        if tlwh[2] * tlwh[3] > 10:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)

        action = 'pending..'
        if len(target.keypoints_list) == 30:
            pts = torch.tensor(target.keypoints_list, dtype=torch.float32)
            pts = pts.numpy()
            out = action_model.predict(pts, frame.shape[:2])
            idx = out[0].argmax()
            prob = out[0].max() * 100
            action = f'{action_model.class_names[idx]}: {prob:.2f}%'
        action_results.append(action)

    return online_tlwhs, online_ids, action_results


def tlwh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] += y[:, 0]
    y[:, 3] += y[:, 1]
    return y


def del_tensor_ele(arr, index_a, index_b):
    arr1 = arr[:index_a]
    arr2 = arr[index_b:]
    return torch.cat((arr1, arr2), dim=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --weights argument removed: only YOLOv11-m-pose.pt is supported in this pipeline
    parser.add_argument('--source', type=str, default='pose_test1.mp4', help='file/folder, 0 for webcam')
    parser.add_argument('--img-size', nargs='+', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-crop', action='store_true', help='save cropped detection boxes')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0 or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', action='store_true', default=False, help='use keypoint labels')
    parser.add_argument('--track_thresh', type=float, default=0.5, help='tracking confidence threshold')
    parser.add_argument('--track_buffer', type=int, default=30, help='the frames for keep lost tracks')
    parser.add_argument('--match_thresh', type=float, default=0.8, help='matching threshold for tracking')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt)
