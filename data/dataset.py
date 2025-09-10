import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class FootDataset(Dataset):
    def __init__(self, root_dir, image_size=512, transforms_=None, unaligned=False):
        self.root_dir = root_dir
        self.unaligned = unaligned
        self.image_size = image_size
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        
        if transforms_ is None:
            self.transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transforms = transforms_

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name).convert('RGB')

        width, height = image.size
        left = image.crop((0, 0, width // 2, height))
        right = image.crop((width // 2, 0, width, height))

        left = self.transforms(left)
        right = self.transforms(right)

        return {'condition': left, 'target': right}

class FootDataset2(Dataset):
    def __init__(self, root_dir, image_size=512, transforms_=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.folder_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                             os.path.isdir(os.path.join(root_dir, f))]

        if transforms_ is None:
            self.transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transforms = transforms_

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]
        files = os.listdir(folder_path)
        condition_img_name = next(f for f in files if f.startswith('original'))
        target_img_name = next(f for f in files if f.startswith('xray'))

        condition_img_path = os.path.join(folder_path, condition_img_name)
        target_img_path = os.path.join(folder_path, target_img_name)

        condition_image = Image.open(condition_img_path).convert('L')
        target_image = Image.open(target_img_path).convert('L')

        condition = self.transforms(condition_image)
        target = self.transforms(target_image)

        return {'condition': condition, 'target': target}


class CTMRI(Dataset):
    def __init__(self, root_dir, image_size=512, transforms_=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.folder_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                             os.path.isdir(os.path.join(root_dir, f))]

        if transforms_ is None:
            self.transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transforms = transforms_

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]
        files = os.listdir(folder_path)
        condition_img_name = next(f for f in files if f.startswith('CT'))
        target_img_name = next(f for f in files if f.startswith('MRI'))

        condition_img_path = os.path.join(folder_path, condition_img_name)
        target_img_path = os.path.join(folder_path, target_img_name)

        condition_image = Image.open(condition_img_path).convert('L')
        target_image = Image.open(target_img_path).convert('L')

        condition = self.transforms(condition_image)
        target = self.transforms(target_image)

        return {'condition': condition, 'target': target}


class MRIPET(Dataset):
    def __init__(self, root_dir, image_size=512, transforms_=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.folder_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                             os.path.isdir(os.path.join(root_dir, f))]

        if transforms_ is None:
            self.transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transforms = transforms_

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]
        files = os.listdir(folder_path)
        condition_img_name = next(f for f in files if f.startswith('MRI'))
        target_img_name = next(f for f in files if f.startswith('PET'))

        condition_img_path = os.path.join(folder_path, condition_img_name)
        target_img_path = os.path.join(folder_path, target_img_name)

        condition_image = Image.open(condition_img_path).convert('L')
        target_image = Image.open(target_img_path).convert('L')

        condition = self.transforms(condition_image)
        target = self.transforms(target_image)

        return {'condition': condition, 'target': target}


class MRISPECT(Dataset):
    def __init__(self, root_dir, image_size=512, transforms_=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.folder_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                             os.path.isdir(os.path.join(root_dir, f))]

        if transforms_ is None:
            self.transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
                ])
        else:
            self.transforms = transforms_

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]
        files = os.listdir(folder_path)
        condition_img_name = next(f for f in files if f.startswith('MRI'))
        target_img_name = next(f for f in files if f.startswith('SPECT'))

        condition_img_path = os.path.join(folder_path, condition_img_name)
        target_img_path = os.path.join(folder_path, target_img_name)

        condition_image = Image.open(condition_img_path).convert('RGB')
        target_image = Image.open(target_img_path).convert('RGB')

        condition = self.transforms(condition_image)
        target = self.transforms(target_image)

        return {'condition': condition, 'target': target}