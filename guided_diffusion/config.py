import argparse, os, sys, datetime, glob, importlib
BRATS_cfg = {
    "resolution": 64,
    "in_channels": 5,
    "out_ch": 5,
    "ch": 128,
    "ch_mult": (1,2,2,2,4),
    "num_res_blocks": 4,
    "attn_resolutions": (16,),
    "dropout": 0.1,
}
IXI_cfg = {
    "resolution": 64,
    "in_channels": 4,
    "out_ch": 4,
    "ch": 128,
    "ch_mult": (1,2,2,2,4),
    "num_res_blocks": 4,
    "attn_resolutions": (16,),
    "dropout": 0.1,
}
model_config_map = {
    "BRATS": BRATS_cfg,
    "IXI": IXI_cfg,
}

diffusion_config = {
    "beta_0": 0.0001,
    "beta_T": 0.02,
    "T": 1000,
}

model_var_type_map = {
    "BRATS": "fixedlarge",
    "IXI": "fixedlarge",
}
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))