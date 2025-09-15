import os, sys, glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import torch


def discover_metals(root):
    root = Path(root)
    metals = []
    for p in sorted([d for d in root.iterdir() if d.is_dir()]):
        if (p/"train/good").exists():
            metals.append(p.name)
    return metals

class MVTecReconDataset(Dataset):
    def __init__(self, root, subdataset, split, img_size=256, augment=False, return_mask=False):
        self.root   = root
        self.sd     = subdataset
        self.split  = split
        self.sd_root = os.path.join(self.root, self.sd)
        
        self.return_mask = return_mask
        if split == "train":
            self.samples = [(p, "good") for p in list_images(os.path.join(self.sd_root,"train","good"))]
        elif split == "test":
            test_dir = os.path.join(self.sd_root, "test")
            defects = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            samples = []
            for cls in defects:
                cdir = os.path.join(test_dir, cls)
                for p in list_images(cdir):
                    samples.append((p, cls))
            self.samples = sorted(samples)
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.tx = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        self.img_size = img_size

    def _read_img(self, path):
        im = Image.open(path).convert("RGB")
        return self.tx(im)

    def _find_mask(self, img_path, cls):
        if cls == "good":  # no defects
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        gt_dir = os.path.join(self.sd_root, "ground_truth", cls)
        if not os.path.exists(gt_dir):
            print(f"[NO GT DIR] {gt_dir}")
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        stem   = Path(img_path).stem  # filename without extension
        ext    = Path(img_path).suffix  # extension, e.g. ".png"

        candidates = []
        candidates += glob.glob(os.path.join(gt_dir, f"{stem}{ext}"))
        candidates += glob.glob(os.path.join(gt_dir, f"{stem}_mask{ext}"))
        candidates += glob.glob(os.path.join(gt_dir, f"{stem}*"))
        # (4) case-insensitive fallback
        if len(candidates) == 0:
            for f in os.listdir(gt_dir):
                if f.lower().startswith(stem.lower()):
                    candidates.append(os.path.join(gt_dir, f))

        if len(candidates) == 0:
            print(f"[NO MASK FOUND] cls={cls}, img={img_path}, stem={stem}, gt_dir={gt_dir}")
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        mask_path = candidates[0]
#         print(f"[MASK FOUND] {img_path}  -->  {mask_path}")

        m = Image.open(mask_path).convert("L")
        m = m.resize((self.img_size, self.img_size), Image.NEAREST)
        m = np.array(m)
        return (m > 0).astype(np.uint8)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, cls = self.samples[i]
        x = self._read_img(p)

        if self.return_mask:
            m = self._find_mask(p, cls)
            m = torch.from_numpy(m).float().unsqueeze(0)
            
            sample = {'img':x,
                      'class':cls,
                      'path':p,
                      'mask':m}
            return sample
        else:
            sample = {'img':x,
                      'class':cls,
                      'path':p}
            
            return sample

def list_images(folder):
    return sorted(glob.glob(os.path.join(folder, "*.png")))
  
class MultiMetalReconDataset(Dataset):
    def __init__(self, root, metals=None, split="train", img_size=256, return_mask=False):
        self.root = Path(root)
        self.metals = metals if metals is not None else discover_metals(root)
        assert len(self.metals) > 0, f"No metals found under {root}"
        self.metal2id = {m:i for i,m in enumerate(self.metals)}
        self.split = split
        self.img_size = img_size
        self.return_mask = return_mask

        tfms = [transforms.Resize((img_size, img_size)),
                transforms.ToTensor()]
        self.tx = transforms.Compose(tfms)

        samples = []
        if split == "train":
            for m in self.metals:
                sd_root = self.root/m
                for p in list_images(sd_root/"train/good"):
                    samples.append((str(p), "good", self.metal2id[m]))
        elif split == "test":
            for m in self.metals:
                sd_root = self.root/m
                test_dir = sd_root/"test"
                if not test_dir.exists(): 
                    continue
                for cls_dir in sorted([d for d in test_dir.iterdir() if d.is_dir()]):
                    cls = cls_dir.name
                    for p in list_images(cls_dir):
                        samples.append((str(p), cls, self.metal2id[m]))
        
        self.samples = samples

    def __len__(self): 
        return len(self.samples)

    def _read_img(self, path):
        im = Image.open(path).convert("RGB")
        return self.tx(im)

    def _find_mask(self, img_path, metal_name, cls):
        if cls == "good":
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        sd_root = self.root/metal_name
        gt_dir = sd_root/f"ground_truth/{cls}"
        if not gt_dir.exists():
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        stem = Path(img_path).stem
        candidates = []
        for e in ("*.png","*.bmp","*.tif","*.jpg","*.jpeg"):
            candidates += glob.glob(str(gt_dir/f"{stem}{Path(img_path).suffix}"))
            candidates += glob.glob(str(gt_dir/f"{stem}_mask.*"))
            candidates += glob.glob(str(gt_dir/f"{stem}.*"))
        if len(candidates)==0:
            candidates = glob.glob(str(gt_dir/f"{stem}*"))

        if len(candidates)==0:
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        m = Image.open(candidates[0]).convert("L")
        m = m.resize((self.img_size, self.img_size), Image.NEAREST)
        m = np.array(m)
        m = (m > 0).astype(np.uint8)
        return m

    def __getitem__(self, i):
        p, cls, metal_id = self.samples[i]
        x = self._read_img(p)
        if self.split == "train":
            return x, torch.as_tensor(metal_id, dtype=torch.long), p
        # test
        if self.return_mask:
            metal_name = self.metals[metal_id]
            m = self._find_mask(p, metal_name, cls)
            m = torch.from_numpy(m).float().unsqueeze(0)
            
            sample = {'img':x,
                      'class':cls,
                      'path':p,
                      'mask':m,
                      'm_id': torch.as_tensor(metal_id, dtype=torch.long)}
            return sample
        else:
            sample = {'img':x,
                      'class':cls,
                      'path':p,
                      'm_id': torch.as_tensor(metal_id, dtype=torch.long)}
            return sample
