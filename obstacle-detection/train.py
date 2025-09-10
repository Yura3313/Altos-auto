import argparse, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from sklearn.metrics import f1_score

start_time = time.time()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="obstacle-detection/images",
                    help="folder with subfolders obstacle/ and no_obstacle/")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--img", type=int, default=[384, 384])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--out", type=str, default=f"obstacle-detection/models/v2.1_{start_time}.pth")
    return ap.parse_args()

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

def weighted_f1_from_confusion(confusion: torch.Tensor):
    tp = confusion.diag().float()
    actual = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
    f1_per_class = torch.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(tp))
    total_support = actual.sum().item()
    if total_support == 0:
        return 0.0
    weighted = (f1_per_class * actual).sum().item() / total_support
    return weighted

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, num_classes, print_freq=10):
    model.train()
    running_loss = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch_idx, (imgs, lbls) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            preds = model(imgs)
            loss = nn.functional.cross_entropy(preds, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # update confusion matrix (move minimal info to CPU)
        pred_labels = preds.argmax(dim=1).cpu().to(torch.long)
        true_labels = lbls.cpu().to(torch.long)

        # quick sanity checks (helpful while debugging)
        if pred_labels.max().item() >= num_classes or true_labels.max().item() >= num_classes:
            print(f"[WARN] label/pred out of range: num_classes={num_classes}, pred_max={pred_labels.max().item()}, true_max={true_labels.max().item()}")
            # fall back to safe per-sample accumulation
            for t, p in zip(true_labels.tolist(), pred_labels.tolist()):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    confusion[t, p] += 1
            # continue to next batch
        else:
            # flattened index MUST be actual * C + predicted
            idx = true_labels * num_classes + pred_labels
            counts = torch.bincount(idx, minlength=num_classes * num_classes)
            confusion += counts.reshape(num_classes, num_classes)

        if print_freq > 0 and batch_idx % print_freq == 0:
            avg_loss = running_loss / batch_idx
            # quick incremental f1 (approx, from current confusion)
            inc_f1 = weighted_f1_from_confusion(confusion)
            print(f"[Epoch {epoch}][{batch_idx}/{len(loader)}] Loss: {avg_loss:.4f}, approx F1: {inc_f1:.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_f1 = weighted_f1_from_confusion(confusion)

    if print_freq > 0:
        print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}, F1 Score: {epoch_f1:.4f}")

    return epoch_loss, epoch_f1

def validate(model, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        with autocast('cuda'):
            for imgs, lbls in loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                preds = model(imgs)
                pred_labels = preds.argmax(dim=1).cpu()
                true_labels = lbls.cpu()
                idx = pred_labels * num_classes + true_labels
                counts = torch.bincount(idx, minlength=num_classes * num_classes)
                confusion += counts.reshape(num_classes, num_classes)
    val_f1 = weighted_f1_from_confusion(confusion)
    return val_f1

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_tf = transforms.Compose([
        transforms.Resize((args.img[0], args.img[1])),
        transforms.ToTensor()
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img[0], args.img[1])),
        transforms.ToTensor()
    ])

    ds_full = datasets.ImageFolder(args.data)
    n_total = len(ds_full)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    ds_train_subset, ds_val_subset = random_split(ds_full, [n_train, n_val], generator=torch.Generator())
    print(f"[INFO] Dataset: {n_total} total, {n_train} train, {n_val} val")

    # Create train and val datasets with transforms
    ds_train = datasets.ImageFolder(args.data, transform=train_tf)
    ds_val = datasets.ImageFolder(args.data, transform=val_tf)

    # Use Subset to split train and val indices
    ds_train = torch.utils.data.Subset(ds_train, ds_train_subset.indices)
    ds_val = torch.utils.data.Subset(ds_val, ds_val_subset.indices)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0, prefetch_factor=2)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=2)
    print(f"[INFO] DataLoaders: {len(dl_train)} train batches, {len(dl_val)} val batches")

    model.to(device)
    print(f"[INFO] Using model: efficientnet_v2_s")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda')

    # compute num_classes from dataset
    num_classes = len(ds_full.classes)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = train_one_epoch(model, dl_train, optimizer, scaler, device, epoch, num_classes)
        val_f1 = validate(model, dl_val, device, num_classes)
        print(f"[INFO] Epoch {epoch} Validation F1 Score: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.out)
            print(f"[INFO] Model saved to {args.out}")

if __name__ == "__main__":
    main()