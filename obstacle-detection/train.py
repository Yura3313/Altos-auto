import argparse, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="obstacle-detection/images",
                    help="folder with subfolders obstacle/ and no_obstacle/")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--img", type=int, default=[384, 384])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--out", type=str, default=default_out)
    ap.add_argument("--model", type=str, default="efficientnet_v2_s", choices=["resnet18", "mobilenetv3_small", 'efficientnet_v2_s'],
                    help="Choose backbone model.")
    return ap.parse_args()

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
default_out = f"obstacle-detection/models/v2_{time.time()}.pth"

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, print_freq=10):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            preds = model(imgs)
            loss = nn.functional.cross_entropy(preds, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_labels.append(lbls.detach().cpu())

    epoch_loss = running_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    epoch_f1 = f1_score(all_labels.numpy(), all_preds.argmax(dim=1).numpy(), average='weighted')

    if print_freq > 0:
        print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}, F1 Score: {epoch_f1:.4f}")

    return epoch_loss, epoch_f1

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
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

        # --- Dataset & Split
    ds_full = datasets.ImageFolder(args.data)
    n_total = len(ds_full)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=torch.Generator())
    print(f"[INFO] Dataset: {n_total} total, {n_train} train, {n_val} val")
    ds_train.dataset.transform = train_tf
    ds_val.dataset.transform = val_tf

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    print(f"[INFO] DataLoaders: {len(dl_train)} train batches, {len(dl_val)} val batches")
    # --- Model, optimizer, scaler
    model.to(device)
    print(f"[INFO] Using model: {args.model}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    # --- Training loop
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = train_one_epoch(model, dl_train, optimizer, scaler, device, epoch)
        # --- Validation
        model.eval()
        with torch.no_grad():
            val_preds, val_labels = [], []
            for imgs, lbls in dl_val:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs)
                val_preds.append(preds.detach().cpu())
                val_labels.append(lbls.detach().cpu())

            val_preds = torch.cat(val_preds)
            val_labels = torch.cat(val_labels)
            val_f1 = f1_score(val_labels.numpy(), val_preds.argmax(dim=1).numpy(), average='weighted')

        print(f"[INFO] Epoch {epoch} Validation F1 Score: {val_f1:.4f}")

        # --- Save model if it has the best F1 score so far
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.out)
            print(f"[INFO] Model saved to {args.out}")

if __name__ == "__main__":
    main()