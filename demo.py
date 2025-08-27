

import io
import os
import pickle
import tarfile
import urllib

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm


PATCH_SIZE = 16
IMAGE_SIZE = 768

# quantization filter for the given patch size
patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

# image resize transform to dimensions divisible by patch size
def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# MODEL_TO_NUM_LAYERS = {
#     MODEL_DINOV3_VITS: 12,
#     MODEL_DINOV3_VITSP: 12,
#     MODEL_DINOV3_VITB: 12,
#     MODEL_DINOV3_VITL: 24,
#     MODEL_DINOV3_VITHP: 32,
#     MODEL_DINOV3_VIT7B: 40,
# }
def cosine_similarity(vec1, vec2):
    """计算余弦相似度（取值范围[-1, 1]）"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # 避免除以零
    return dot_product / (norm_vec1 * norm_vec2)


patch_mask_values = []
patch_features = []

DINOV3_LOCATION = "./dinov3_vits16plus"
MODEL_NAME ="dinov3_vits16plus"
#MODEL_NAME = "dinov3_convnext_tiny"

n_layers = 12 #MODEL_TO_NUM_LAYERS[MODEL_NAME]

model = torch.hub.load(
    repo_or_dir=DINOV3_LOCATION,
    model=MODEL_NAME,
    source="local",
)
print(dir(model))
print(model)
image = Image.open("3.jpg")
image1 = Image.open("5.jpeg")
import time
with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.float32):

        # processing image
        image = image.convert('RGB')
        image_resized = resize_transform(image)
        image_resized = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        image_resized = image_resized.unsqueeze(0) #.cuda()
        print('inputs shape: ', image_resized.shape)
        # inputs = processor(images=img1, return_tensors="pt")
        outputs1 = model.forward(image_resized)


        for _ in range(1):
            t0 = time.time()
            image1 = image1.convert('RGB')
            image_resized = resize_transform(image1)
            image_resized = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            image_resized = image_resized.unsqueeze(0)  # .cuda()
            print('inputs shape: ', image_resized.shape)
            # inputs = processor(images=img1, return_tensors="pt")
            outputs2 = model.forward(image_resized)
            vec1 = outputs1.numpy().squeeze(0)
            vec2 = outputs2.numpy().squeeze(0)
            res = cosine_similarity(vec1, vec2)
            print(res)
            print("use time: ", time.time()-t0)
        # feats = model.get_intermediate_layers(image_resized, n=range(n_layers), reshape=True, norm=True)
        # # print("feats: ", feats[-1].shape)
        # print(dir(feats))
        # for ii in feats:
        #     print(ii.shape)
        # dim = feats[-1].shape[1]
        # #print("dim: ", dim)
        # #patch_features.append(feats[-1].squeeze().view(dim, -1).permute(1,0).detach().cpu())
        # feas =  feats[-1].squeeze().detach().cpu()
        #print(feas.shape)





