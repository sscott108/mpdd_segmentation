import argparse, os, json
from torch.utils.data import DataLoader
from datasets import MultiMetalReconDataset,MVTecReconDataset
from models import get_model
from loss_utils import get_loss
from train import train_model
from test import test

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--metal", type=str, required=True, help="Which single metal to run Plain U-Net on")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    #single unet task on 1 metal, reconstruct masks to compare with anomaly test images
    print(f"\n[Plain U-Net] Training on single metal: {args.metal}")
    plain_train_ds = MVTecReconDataset(args.data_root, args.metal, split="train", img_size=args.img_size)
    
    plain_train_dl = DataLoader(plain_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
#     print(next(iter(plain_train_dl)))
    
    plain_test_ds  = MVTecReconDataset(args.data_root, args.metal, split="test", img_size=args.img_size, return_mask=True)
    plain_test_dl  = DataLoader(plain_test_ds, batch_size=1, shuffle=False)    
    plain_model = get_model("plain_unet")
    plain_loss  = get_loss("l1_edge")
    plain_model = train_model(plain_model, plain_train_dl, plain_loss, args.epochs, args.lr, args.device, use_shared=False)
    plain_metrics = test(plain_model, plain_test_dl, args.device, os.path.join(args.save_dir, f"{args.metal}_plain"), use_shared=False)

    #shared unet task on all metals to compare
    print("\n[Shared U-Net] Training on all metals...")
#     break
    shared_train_ds = MultiMetalReconDataset(args.data_root, split="train", img_size=args.img_size)
    shared_train_dl = DataLoader(shared_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    shared_test_ds  = MultiMetalReconDataset(args.data_root, metals=[args.metal], split="test", img_size=args.img_size, return_mask=True)
    shared_test_dl  = DataLoader(shared_test_ds, batch_size=1, shuffle=False)

    shared_model = get_model("shared_unet", num_metals=len(shared_train_ds.metals))
    shared_loss  = get_loss("l1_edge")
    shared_model = train_model(shared_model, shared_train_dl, shared_loss, args.epochs, args.lr, args.device, use_shared=True)
    shared_metrics = test(shared_model, shared_test_dl, args.device, os.path.join(args.save_dir, f"{args.metal}_shared"), use_shared=True)


    comparison = {
        "metal": args.metal,
        "plain_unet": plain_metrics,
        "shared_unet": shared_metrics}
    with open(os.path.join(args.save_dir, f"{args.metal}_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n==== FINAL COMPARISON ====")
    print(json.dumps(comparison, indent=2))
