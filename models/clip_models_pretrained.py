from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch
from .loralib.utils import apply_lora, mark_only_lora_as_trainable



class CLIPModel_pretrained(nn.Module):
    def __init__(self, opt, num_classes=1):
        super(CLIPModel_pretrained, self).__init__()

        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.clip_model_frozen, self.preprocess = clip.load(self.opt.backbone, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 

        self.img_frozen_fc = nn.Linear(768, num_classes)
        

    def forward(self, x):
        with torch.no_grad():
            img_frozen_features = self.clip_model_frozen.encode_image(x)
        img_frozen_logits = self.img_frozen_fc(img_frozen_features)

        return img_frozen_logits, img_frozen_features