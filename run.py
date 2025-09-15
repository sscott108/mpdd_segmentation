import argparse, os, json, csv
from torch.utils.data import DataLoader
from datasets import MultiMetalReconDataset, MVTecReconDataset, discover_metals
from models import get_model
from loss_utils import get_loss
from train import train_model
from test import test
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--save_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Shared model trained once on ALL metals ----
    print("\n[Shared U-Net] Training on all metals...")
    shared_train_ds = MultiMetalReconDataset(args.data_root, split="train", img_size=args.img_size)
    shared_train_dl = DataLoader(shared_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    shared_model = get_model("shared_unet", num_metals=len(shared_train_ds.metals))
    shared_loss  = get_loss("l1_edge")
    shared_model = train_model(shared_model, shared_train_dl, shared_loss,
                               args.epochs, args.lr, args.device, use_shared=True)

    # ---- Iterate over metals ----
    results = []
    metals = discover_metals(args.data_root)

    for metal in metals:
        print(f"\n[Plain U-Net] Training on single metal: {metal}")
        plain_train_ds = MVTecReconDataset(args.data_root, metal, split="train", img_size=args.img_size)
        plain_train_dl = DataLoader(plain_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        plain_test_ds  = MVTecReconDataset(args.data_root, metal, split="test", img_size=args.img_size, return_mask=True)
        plain_test_dl  = DataLoader(plain_test_ds, batch_size=1, shuffle=False)

        plain_model = get_model("plain_unet")
        plain_loss  = get_loss("l1_edge")
        plain_model = train_model(plain_model, plain_train_dl, plain_loss,
                                  args.epochs, args.lr, args.device, use_shared=False)
        plain_metrics = test(plain_model, plain_test_dl, args.device,
                             os.path.join(args.save_dir, f"{metal}_plain"), use_shared=False)

        print(f"\n[Shared U-Net] Testing on metal: {metal}")
        shared_test_ds = MultiMetalReconDataset(args.data_root, metals=[metal], split="test",
                                                img_size=args.img_size, return_mask=True)
        shared_test_dl = DataLoader(shared_test_ds, batch_size=1, shuffle=False)
        shared_metrics = test(shared_model, shared_test_dl, args.device,
                              os.path.join(args.save_dir, f"{metal}_shared"), use_shared=True)

        comparison = {
            "metal": metal,
            "plain_unet": plain_metrics,
            "shared_unet": shared_metrics
        }
        results.append(comparison)

        with open(os.path.join(args.save_dir, f"{metal}_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)

    # ---- Save aggregated results ----
    df_rows = []
    for r in results:
        df_rows.append({
            "metal": r["metal"],
            "plain_pixel_AUROC": r["plain_unet"]["pixel_AUROC"],
            "plain_best_Dice": r["plain_unet"]["best_Dice"],
            "shared_pixel_AUROC": r["shared_unet"]["pixel_AUROC"],
            "shared_best_Dice": r["shared_unet"]["best_Dice"],
        })
    df = pd.DataFrame(df_rows)
    df.to_csv(os.path.join(args.save_dir, "all_results.csv"), index=False)
    print("\n==== FINAL TABLE ====")
    print(df)

    # Optionally plot table as heatmap
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        plt.savefig(os.path.join(args.save_dir, "results_table.png"))
        print(f"Saved table plot to {os.path.join(args.save_dir, 'results_table.png')}")
    except ImportError:
        print("matplotlib not installed, skipping table plot")
