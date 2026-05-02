from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch
from .loralib.utils import apply_lora, mark_only_lora_as_trainable
from models.clip_models_pretrained import CLIPModel_pretrained



class CLIPModel(nn.Module):
    def __init__(self, opt, num_classes=1):
        super(CLIPModel, self).__init__()

        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.clip_model, self.preprocess = clip.load(self.opt.backbone, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class
        self.clip_model_frozen = CLIPModel_pretrained(self.opt)
        self.clip_model_frozen.load_state_dict(torch.load("./checkpoints/model_epoch_step1.pth"), strict=True)
        list_lora_layers = apply_lora(self.opt, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)

        self.text_fc = nn.Linear(768, num_classes)
        self.img_fc = nn.Linear(768, num_classes)

        for p in self.clip_model_frozen.img_frozen_fc.parameters(): p.requires_grad = False
        

    def forward(self, x, text=[]):
        if text == []:
            with torch.no_grad():
                features = self.clip_model.encode_image(x)
                img_logits = self.img_fc(features)
                return img_logits

        features = self.clip_model.encode_image(x)
        with torch.no_grad():
            img_frozen_features = self.clip_model_frozen.clip_model_frozen.encode_image(x)

            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)

        img_frozen_features = img_frozen_features.detach().requires_grad_()
        text_features = text_features.detach().requires_grad_()

        img_logits = self.img_fc(features)
        img_frozen_logits = self.clip_model_frozen.img_frozen_fc(img_frozen_features)
        text_logits = self.text_fc(text_features)

        return img_logits, features, img_frozen_logits, img_frozen_features, text_logits, text_features