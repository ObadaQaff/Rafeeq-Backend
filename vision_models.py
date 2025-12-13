import torch

depth_model = torch.hub.load(
    "intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True
)
depth_model.eval()

depth_transform = torch.hub.load(
    "intel-isl/MiDaS", "transforms", trust_repo=True
).dpt_transform
