import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.transforms import *

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

########################################################################
#                           Augmentation                               #
#######################################################################
class BaseAugmentation:
    '''
    얼굴 부분만을 효율적으로 crop하기 이해  
    원하는 size+32로 resize 해준 후,Centercrop  
    '''
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize + 32 , Image.BILINEAR),
            CenterCrop(resize),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
    
    
class AffineAugmentation:
    '''BASE + RandomAffine'''
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(256 , Image.BILINEAR),
            CenterCrop(resize),
            RandomAffine(10 , (0,0.1) , (1,1.2) , (0,0.1) , Image.BILINEAR ),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)

class PerspectiveAugmentation:
    '''BASE + RandomPerspective'''
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(256 , Image.BILINEAR),
            CenterCrop(resize),
            RandomPerspective(distortion_scale=0.1, p=1, interpolation= Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)

class RandomAugmentation:
    '''BASE + RandomCrop'''
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(256 , Image.BILINEAR),
            CenterCrop(resize),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        self.transform.transforms.insert(0, RandAugment(2, 8))
        return self.transform(image)
    
    
class CustomRandomCrop(torch.nn.Module):
    '''
    Custom transform
    center를 중심으로 랜덤으로 이동해 crop
    '''
    def __init__(self, height, width):
        self.crop_height = height
        self.crop_width = width

    def forward(self, img):
        return self.get_center_crop_coords(img,self.crop_height,self.crop_width)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
    
    def get_center_crop_coords(img , crop_height: int, crop_width: int):
        transformed_img = img.copy()
        height = img.shape[0]
        width = img.shape[1]
        delta = random.uniform(-0.1,0.1)
        y1 = int(np.round((1 + delta) * (height - crop_height) // 2))
        y2 = y1 + crop_height
        x1 = int(np.round((1 + delta) * (width - crop_width) // 2))
        x2 = x1 + crop_width
        return transformed_img[y1:y2 , x1:x2]
    


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CustomRandomCrop(resize,resize),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)

########################################################################
#                               DATSET                                 #
#######################################################################
class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    class GenderLabels:
        male = 0
        female = 1

    class AgeGroup():
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 58 else 2

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2 , age_filter = True):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.age_filter  = age_filter
        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = getattr(self.GenderLabels, gender)
                age_label = self.AgeGroup().map_label(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]

    
class MultiBranchDataset(MaskSplitByProfileDataset):
    '''
    Multi-Branch 모델을 위해서 각각 변수 (mask, gender,age)에 맞춰 return
    나머지 기능은 basecode의 MaskSplitByProfileDataset 상속받음
    '''
    num_classes = 3
    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        image_transform = self.transform(image)
        return image_transform, (mask_label , gender_label ,age_label)    

class MaskDataset(MaskSplitByProfileDataset):
    '''
    Mask label만 return 
    나머지 기능은 basecode의 MaskSplitByProfileDataset 상속받음
    '''
    num_classes = 3
    def __getitem__(self, index):
        super().__getitem__(index)

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)

        image_transform = self.transform(image)
        return image_transform, mask_label

class AgeDataset(MaskSplitByProfileDataset):
    '''
    Age label만 return 
    나머지 기능은 basecode의 MaskSplitByProfileDataset 상속받음
    '''
    num_classes = 3
    def __getitem__(self, index):
        super().__getitem__(index)

        image = self.read_image(index)
        age_label = self.get_age_label(index)

        image_transform = self.transform(image)
        return image_transform, age_label

class GenderDataset(MaskSplitByProfileDataset):
    '''
    Gender label만 return 
    나머지 기능은 basecode의 MaskSplitByProfileDataset 상속받음
    '''
    num_classes = 2
    def __getitem__(self, index):
        super().__getitem__(index)

        image = self.read_image(index)
        gender_label = self.get_gender_label(index)

        image_transform = self.transform(image)
        return image_transform, gender_label

class Age_Gender_Dataset(MaskSplitByProfileDataset):
    '''
    Mask , Age & Gender 모델을 따로 만들기 위해 Age & Gender label을 계산해 return
    나머지 기능은 basecode의 MaskSplitByProfileDataset 상속받음
    '''
    num_classes = 6
    def __getitem__(self, index):
        super().__getitem__(index)

        image = self.read_image(index)
        mask_label = 0
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        image_transform = self.transform(image)
        return image_transform, multi_class_label

    
class KfoldDataset(MultiBranchDataset):
    
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2 , k=0):
        self.indices = defaultdict(list)
        self.k = k
        super().__init__(data_dir, mean, std, val_ratio)
    
    
    @staticmethod
    def _split_profile(profiles, val_ratio , k):
        length = len(profiles)
        n_val = int(length * val_ratio)
        
        
        val_indices = set(range(int(k*length/5) ,int((k+1)*length/5)))
        print(f"val_indices_ {k}:{int(k*length/5)} ~ {int((k+1)*length/5)}")
        train_indices = set(range(length)) - val_indices
        
        return {
            "train": train_indices,
            "val": val_indices
        }
    
    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio , self.k)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1    
    
class TestDataset(data.Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize((resize,resize), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
