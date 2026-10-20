import numpy as np
import torch
import cv2
import os
import argparse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM2 with multi-image batching")
    parser.add_argument("--data_dir", type=str, default="LabPicsV1/Simple/Train", help="Path to dataset (should contain Image/ and Instance/)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu"], help="Device to use")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "large"], help="Model size")
    parser.add_argument("--save_every", type=int, default=1, help="Save model every N epochs")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--exp_name", type=str, default="training_current_exp", help="Experiment name for checkpoint subdirectory")
    return parser.parse_args()

def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg in ["cuda", "gpu"]:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_model_config_and_ckpt(model_size):
    if model_size == "small":
        return "sam2_hiera_s.yaml", "sam2_hiera_small.pt"
    elif model_size == "large":
        return "sam2_hiera_l.yaml", "sam2_hiera_large.pt"
    else:
        raise ValueError("Invalid model_size")

def load_data(data_dir, split="Train"):
    image_dir = os.path.join(data_dir, f"{split}/Image")
    mask_dir = os.path.join(data_dir, f"{split}/Instance")
    data = []
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"⚠️  {split} split not found at {image_dir} and {mask_dir}.")
        return data
    for name in os.listdir(image_dir):
        if name.lower().endswith(('.png', '.jpg', '.jpeg')):
            data.append({
                "image": os.path.join(image_dir, name),
                "annotation": os.path.join(mask_dir, name[:-4] + ".png")
            })
    print(f"📦 Found {len(data)} image/mask pairs for {split.lower()}.")
    return data

def read_single(data):
    ent = data[np.random.randint(len(data))]
    Img = cv2.imread(ent["image"])[..., ::-1]
    ann_map = cv2.imread(ent["annotation"])
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    if Img.shape[0] < 1024:
        Img = np.concatenate([Img, np.zeros([1024 - Img.shape[0], Img.shape[1], 3], dtype=np.uint8)], axis=0)
        ann_map = np.concatenate([ann_map, np.zeros([1024 - ann_map.shape[0], ann_map.shape[1], 3], dtype=np.uint8)], axis=0)
    if Img.shape[1] < 1024:
        Img = np.concatenate([Img, np.zeros([Img.shape[0], 1024 - Img.shape[1], 3], dtype=np.uint8)], axis=1)
        ann_map = np.concatenate([ann_map, np.zeros([ann_map.shape[0], 1024 - ann_map.shape[1], 3], dtype=np.uint8)], axis=1)
    mat_map = ann_map[:, :, 0]
    ves_map = ann_map[:, :, 2]
    mat_map[mat_map == 0] = ves_map[mat_map == 0] * (mat_map.max() + 1)
    inds = np.unique(mat_map)[1:]
    if len(inds) > 0:
        ind = inds[np.random.randint(len(inds))]
    else:
        return read_single(data)
    mask = (mat_map == ind).astype(np.uint8)
    coords = np.argwhere(mask > 0)
    yx = np.array(coords[np.random.randint(len(coords))])
    return Img, mask, [[yx[1], yx[0]]]

def read_batch(data, batch_size=4):
    limage, lmask, linput_point = [], [], []
    for i in range(batch_size):
        image, mask, input_point = read_single(data)
        limage.append(image)
        lmask.append(mask)
        linput_point.append(input_point)
    return limage, np.array(lmask), np.array(linput_point), np.ones([batch_size, 1])

def compute_metrics(gt_mask, prd_mask):
    gt = gt_mask.flatten()
    prd = (prd_mask > 0.5).float().flatten()
    correct = (gt == prd).sum().item()
    total = gt.numel()
    pixel_acc = correct / total if total > 0 else 0.0
    tp = ((gt == 1) & (prd == 1)).sum().item()
    fp = ((gt == 0) & (prd == 1)).sum().item()
    fn = ((gt == 1) & (prd == 0)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return pixel_acc, precision, recall, f1

class ModelCheckpoint:
    def __init__(self, save_every=1, prefix="model_epoch_", verbose=True, model_size="small", output_dir="checkpoints"):
        self.save_every = save_every
        self.prefix = prefix
        self.verbose = verbose
        self.model_size = model_size
        self.output_dir = output_dir

    def __call__(self, model, epoch, total_epochs):
        if epoch % self.save_every == 0 or epoch == total_epochs:
            filename = os.path.join(self.output_dir, f"{self.prefix}{self.model_size}_epoch{epoch}.pth")
            torch.save(model, filename)
            if self.verbose:
                print(f"💾 [Callback] Full model saved at epoch {epoch} as {filename}")

def main():
    args = parse_args()
    device = get_device(args.device)
    print("=" * 60)
    print(f"🚀 Using device: {device}")
    print("=" * 60)
    model_cfg, sam2_checkpoint = get_model_config_and_ckpt(args.model_size)
    print(f"🔄 Loading model and weights: {sam2_checkpoint} ({args.model_size})")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("✅ Model loaded and ready.")

    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    predictor.model.image_encoder.train(True)
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Create experiment subdirectory for checkpoints
    exp_ckpt_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(save_every=args.save_every, prefix="model_epoch_", verbose=True, model_size=args.model_size, output_dir=exp_ckpt_dir)

    train_data = load_data(args.data_dir, split="Train")
    val_data = load_data(args.data_dir, split="Val")
    test_data = load_data(args.data_dir, split="Test")
    print("=" * 60)
    print("🏁 Starting training...")
    print("=" * 60)

    from sklearn.metrics import confusion_matrix

    def evaluate_on_split(data, split_name):
        if not data:
            print(f"⚠️  No data for {split_name} split, skipping evaluation.")
            return
        print(f"\n{'='*30} 🔎 {split_name.upper()} SET EVALUATION {'='*30}")
        total_loss, total_iou, total_acc, total_prec, total_rec, total_f1, n = 0, 0, 0, 0, 0, 0, 0
        all_gt = []
        all_pred = []
        for i in range(len(data)):
            image, mask, input_point = read_single(data)
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts([input_point], np.ones([1,1]), box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            gt_mask = torch.tensor(mask.astype(np.float32), device=device)
            prd_mask = torch.sigmoid(prd_masks[:, 0].to(device))
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
            inter = (gt_mask * (prd_mask > 0.5)).sum()
            iou = inter / (gt_mask.sum() + (prd_mask > 0.5).sum() - inter)
            batch_acc, batch_prec, batch_rec, batch_f1 = compute_metrics(gt_mask.cpu(), prd_mask.cpu())
            total_loss += seg_loss.item()
            total_iou += iou.item()
            total_acc += batch_acc
            total_prec += batch_prec
            total_rec += batch_rec
            total_f1 += batch_f1
            n += 1
            all_gt.extend(gt_mask.cpu().numpy().flatten())
            all_pred.extend((prd_mask.cpu().detach().numpy().flatten() > 0.5).astype(int))
        print(f"\n{'-'*20} 📊 {split_name} Metrics 📊 {'-'*20}")
        print(f"Loss:      {total_loss/n:.4f}")
        print(f"IOU:       {total_iou/n:.4f}")
        print(f"Accuracy:  {total_acc/n:.4f}")
        print(f"Precision: {total_prec/n:.4f}")
        print(f"Recall:    {total_rec/n:.4f}")
        print(f"F1 Score:  {total_f1/n:.4f}")
        # Confusion matrix
        cm = confusion_matrix(all_gt, all_pred, labels=[0, 1])
        print(f"\nConfusion Matrix (rows: true, cols: pred):\n{cm}")
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        total = tn + fp + fn + tp
        print(f"Summary: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        if total > 0:
            print(f"Normalized: TP={tp/total:.4%}, FP={fp/total:.4%}, TN={tn/total:.4%}, FN={fn/total:.4%}")
        print("Note: Confusion matrix is at the pixel level. For instance segmentation, consider using mAP or other instance metrics.")
        print("="*70)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_acc = 0.0
        epoch_prec = 0.0
        epoch_rec = 0.0
        epoch_f1 = 0.0
        num_batches = 0

        print(f"\n{'='*25} 🌟 Epoch {epoch}/{args.epochs} 🌟 {'='*25}")
        for step in range(args.steps_per_epoch):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                image, mask, input_point, input_label = read_batch(train_data, batch_size=args.batch_size)
                if mask.shape[0] == 0:
                    print(f"  [Batch {step+1}] Skipped empty mask batch.")
                    continue
                predictor.set_image_batch(image)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"],
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                gt_mask = torch.tensor(mask.astype(np.float32), device=device)
                prd_mask = torch.sigmoid(prd_masks[:, 0].to(device))
                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                score_loss = torch.abs(prd_scores[:, 0].to(device) - iou).mean()
                loss = seg_loss + score_loss * 0.05

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_iou = np.mean(iou.cpu().detach().numpy())
            batch_loss = loss.item()
            batch_acc, batch_prec, batch_rec, batch_f1 = compute_metrics(gt_mask.cpu(), prd_mask.cpu())

            epoch_loss += batch_loss
            epoch_iou += batch_iou
            epoch_acc += batch_acc
            epoch_prec += batch_prec
            epoch_rec += batch_rec
            epoch_f1 += batch_f1
            num_batches += 1

            print(f"  [Batch {step+1:03d}/{args.steps_per_epoch}] "
                  f"Loss: {batch_loss:.4f} | IOU: {batch_iou:.4f} | "
                  f"Acc: {batch_acc:.4f} | Prec: {batch_prec:.4f} | "
                  f"Rec: {batch_rec:.4f} | F1: {batch_f1:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_iou = epoch_iou / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        avg_prec = epoch_prec / max(num_batches, 1)
        avg_rec = epoch_rec / max(num_batches, 1)
        avg_f1 = epoch_f1 / max(num_batches, 1)
        print(f"\n{'-'*20} 📝 Epoch {epoch} Summary 📝 {'-'*20}")
        print(f"Avg Loss: {avg_loss:.4f} | Avg IOU: {avg_iou:.4f} | "
              f"Avg Acc: {avg_acc:.4f} | Avg Prec: {avg_prec:.4f} | "
              f"Avg Rec: {avg_rec:.4f} | Avg F1: {avg_f1:.4f}")
        print("-"*60)

        # Validation after each epoch
        if val_data:
            evaluate_on_split(val_data, "Val")

        checkpoint_callback(predictor.model, epoch, args.epochs)

    print("=" * 60)
    print("🎉 Training complete. All epochs finished! 🎉")
    print("=" * 60)

    # Save final model
    final_model_path = os.path.join(exp_ckpt_dir, f"model_full_{args.model_size}_final.pth")
    torch.save(predictor.model, final_model_path)
    print(f"✅ Final full model saved as {final_model_path}")

    # Evaluate on test set ONCE after training
    if test_data:
        evaluate_on_split(test_data, "Test")

if __name__ == "__main__":
    main()

# ==========================
# Example CLI command:
#
# Train with large model on GPU:
# python TRAIN_multi_image_batch.py --device auto --model_size large --epochs 1 --batch_size 4 --steps_per_epoch 100 --data_dir LabPicsV1/Simple/Train --output_dir checkpoints/ --exp_name training_current_exp
#
# Train with small model on CPU:
# python TRAIN_multi_image_batch.py --device cpu --model_size small --epochs 1 --batch_size 2 --steps_per_epoch 100 --data_dir LabPicsV1/Simple/Train --output_dir checkpoints/ --exp_name training_current_exp
#
# (Checkpoints and final model will be saved in checkpoints/training_current_exp/)
# Example final model path: checkpoints/training_current_exp/model_full_large_final.pth
#
# (Adjust --device, --model_size, --epochs, --batch_size, --steps_per_epoch, --data_dir, --output_dir, --exp_name as needed)
# ==========================
