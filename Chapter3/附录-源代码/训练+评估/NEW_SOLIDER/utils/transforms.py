import random
from copy import deepcopy
from torchvision.transforms import functional as F

# 组合多个数据变换操作的类
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# 随机水平翻转变换操作
class RandomHorizontalFlip:
    def __init__(self, prob=0.5):  # 概率默认为0.5
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 调整目标中的边界框的坐标
            target["boxes"] = bbox
        return image, target

# [0, 255] 范围转换为 [0, 1] 范围的张量
class ToTensor:
    def __call__(self, image, target):
        # convert [0, 255] to [0, 1]
        image = F.to_tensor(image)
        return image, target

# transform：图像数据集预处理
def build_transforms(is_train):
    transforms = []
    transforms.append(ToTensor())
    if is_train:
        transforms.append(RandomHorizontalFlip())
    return Compose(transforms)

def mixup_data(images, alpha=0.8):
    if alpha > 0. and alpha < 1.:
        lam = random.uniform(alpha, 1)
    else:
        lam = 1.

    batch_size = len(images)
    min_x = 9999
    min_y = 9999
    for i in range(batch_size):
        min_x = min(min_x, images[i].shape[1])
        min_y = min(min_y, images[i].shape[2])

    shuffle_images = deepcopy(images)
    random.shuffle(shuffle_images)
    mixed_images = deepcopy(images)
    for i in range(batch_size):
        mixed_images[i][:, :min_x, :min_y] = lam * images[i][:, :min_x, :min_y] + (1 - lam) * shuffle_images[i][:, :min_x, :min_y]

    return mixed_images