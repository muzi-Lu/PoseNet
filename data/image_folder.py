###### ImageFolder 是干什么的我都不知道 #####

from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

##### 没懂什么意思
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#####
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = osp.join(path, fname)
                images.append(path)
    return sorted(images)

def default_loader(path):
    return Image.open(path).convet('RGB')

class ImageFolder(Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        # Not Finished
        pass

    def __len__(self):
        return len(self.imgs)
