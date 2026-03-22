import argparse

import torch
from torchvision import transforms
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA

from sprim.utils.dino_extractor import ViTExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)

args = parser.parse_args()


img = Image.open(args.image).convert("RGB")

extraction_shape = (
    img.height // 8 * 8,
    img.width // 8 * 8,
)
prep = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop(extraction_shape),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.Resize((extraction_shape[0] // 2, extraction_shape[1] // 2)),
    ]
)


images = [prep(img)]
preproc_image_lst = torch.stack(images, dim=0).to("cuda")

extractor = ViTExtractor(
    model_type="dino_vitb8",
    stride=8,
)

with torch.no_grad():
    descriptors = extractor.extract_descriptors(
        preproc_image_lst,
        [11],  # 11 for vit-b and vit-s, 23 for vit-l
        "key",
        include_cls=False,
    )
descriptors = descriptors.reshape(
    descriptors.shape[0],
    extractor.num_patches[0],
    extractor.num_patches[1],
    -1,
).squeeze()


descriptors = descriptors.cpu().detach().numpy()

pca = PCA(n_components=3)
feature_maps_pca = pca.fit_transform(descriptors.reshape(-1, descriptors.shape[-1]))
pca_features_min = feature_maps_pca.min(axis=(0, 1))
pca_features_max = feature_maps_pca.max(axis=(0, 1))
pca_features = (feature_maps_pca - pca_features_min) / (
    pca_features_max - pca_features_min
)

pca_features = pca_features.reshape(descriptors.shape[0], descriptors.shape[1], 3)
Image.fromarray((pca_features * 255).astype(np.uint8)).save("flower_pca.png")
