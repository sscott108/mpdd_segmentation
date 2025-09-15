import os, cv2, json, csv
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

def train_model(model, loader, loss_fn, epochs=50, lr=1e-3, device="cpu", use_shared=False):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs+1):
        losses=[]
        for batch in tqdm(loader, desc=f"Epoch {ep}/{epochs}", leave=False):
            if use_shared:
                x        = batch['img'].to(device)            
                metal_id = batch['m_id'].to(device)
                
                xh       = model(x, metal_ids=metal_id)
            else:
                x  = batch['img'].to(device)
                xh = model(x)

            loss = loss_fn(x, xh)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[Epoch {ep}] loss={np.mean(losses):.5f}")
    return model