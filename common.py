"""
Shared constants + image transforms.
Training, evaluation, the web app and predict.py ALL import from here, so the
preprocessing used at inference is guaranteed to match training.
"""
from torchvision import transforms

MODEL_ZOO = {
    "resnet50": "resnet50", "densenet": "densenet121", "efficientnet": "efficientnet_b0",
    "vit": "vit_base_patch16_224", "deit": "deit_small_patch16_224",
    "swin": "swin_tiny_patch4_window7_224", "convnext": "convnext_tiny",
    "maxvit": "maxvit_tiny_tf_224",
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet stats (models are ImageNet-pretrained)


def _gray(gray):
    # X-rays / MRI are single-channel; replicate to 3 channels for ImageNet models.
    return [transforms.Grayscale(num_output_channels=3)] if gray else []


def train_transform(img_size=224, gray=True):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, scale=(0.9, 1.1)),
        *_gray(gray),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])


def eval_transform(img_size=224, gray=True):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        *_gray(gray),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
