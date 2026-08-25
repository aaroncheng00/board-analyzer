import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
])

PATH_DATA_TRAIN = "data/train/cells"
PATH_DATA_VAL = "data/val/cells"

train_dataset = datasets.ImageFolder(PATH_DATA_TRAIN, transform=transform)
val_dataset = datasets.ImageFolder(PATH_DATA_VAL, transform=transform)

print(train_dataset.class_to_idx) # confirm data loaded correctly

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

imshow(torchvision.utils.make_grid(images))