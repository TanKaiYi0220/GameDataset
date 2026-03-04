import torch

import sys
sys.path.append('models/IFRNet')
from tqdm import tqdm

# from models.IFRNet import Model
from models.IFRNet import Model as IFRNetModel
from models.IFRNet_Residual import Model as IFRNetResidualModel

from thop import profile

model = IFRNetModel()  # 你的模型
model.eval()

H, W = 1280, 720

class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, I0, I1, t):
        # 這裡改成你的推論入口
        return self.model.inference(I0, I1, t)   # 或 self.model.forward_test(...)

model.eval().cuda()
wrap = InferenceWrapper(model).eval().cuda()

I0 = torch.randn(1, 3, H, W).cuda()
I1 = torch.randn(1, 3, H, W).cuda()
t  = torch.tensor([0.5], device="cuda").view(1, 1)

with torch.no_grad():
    macs, params = profile(wrap, inputs=(I0, I1, t), verbose=False)

print('MACs: {:.3f}T'.format(macs / 1e12))
print('Params: {:.3f}M'.format(params / 1e6))

model = IFRNetResidualModel()  # 你的模型
model.eval()

H, W = 1280, 720

class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, I0, I1, t, f0, f1):
        # 這裡改成你的推論入口
        return self.model.inference(I0, I1, t, init_flow0_full=f0, init_flow1_full=f1)   # 或 self.model.forward_test(...)

model.eval().cuda()
wrap = InferenceWrapper(model).eval().cuda()

I0 = torch.randn(1, 3, H, W).cuda()
I1 = torch.randn(1, 3, H, W).cuda()
F0 = torch.randn(1, 2, H, W).cuda()
F1 = torch.randn(1, 2, H, W).cuda()
t  = torch.tensor([0.5], device="cuda").view(1, 1)

with torch.no_grad():
    macs, params = profile(wrap, inputs=(I0, I1, t, F0, F1), verbose=False)

print('MACs: {:.3f}T'.format(macs / 1e12))
print('Params: {:.3f}M'.format(params / 1e6))