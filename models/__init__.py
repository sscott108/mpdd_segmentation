from .unet_shared_ import SharedUNetAE
from .unet_plain_ import UNetAE

def get_model(name, **kwargs):
    if name == "shared_unet":
        return SharedUNetAE(in_ch=3, base=64, out_ch=3, **kwargs)
    elif name == "plain_unet":
        return UNetAE(in_ch=3, base=32)
    else:
        raise ValueError(f"Unknown model: {name}")
