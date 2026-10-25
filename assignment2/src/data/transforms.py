#transforms.py
'''
RGB transforms for 30VNFoods.

Normalization uses the ImageNet constants that belong to
torchvision ResNet50_Weights.IMAGENET1K_V2; they are not computed from our data.
Evaluation preprocessing matches IMAGENET1K_V2.transforms():
resize the shorter side to 232, center-crop 224.
Augmentation is applied to training images only.

Output: float tensor [3, 224, 224].
'''

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SIZE = 232


def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transforms(image_size=224):
    return get_val_transforms(image_size)
