import torch
import numbers
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

from imgaug import augmenters as iaa
import imgaug

def generate_vote_from_pts(label, pts, mask):
    # print("pts[1] = {}".format(pts[1]))
    height, width = label.shape
    pt2d = pts[label]
    pt2d = pt2d.reshape(height, width, 2)

    vertex_2d = np.zeros((height, width, 2), dtype=np.float64)
    vertex_2d[..., 0] = np.tile(np.arange(width, dtype=np.float64), height).reshape(height, width)
    vertex_2d[..., 1] = np.tile(np.arange(height, dtype=np.float64), width).reshape(width, height).transpose(1, 0)

    vertex_2d = pt2d - vertex_2d
            
    length = np.sqrt(vertex_2d[..., 0] ** 2 + vertex_2d[..., 1] ** 2 ) + 1e-8
    vertex_2d[..., 0] = vertex_2d[..., 0] / length
    vertex_2d[..., 1] = vertex_2d[..., 1] / length
    vertex_2d[label==0] = 0
    vertex_2d[mask==0] = 0
    
    return vertex_2d

def affine_aug(img, label, id2centers, rvec, tvec, k_matrix, aug):
    
    if aug:
        trans_x = random.uniform(-0.2,0.2)
        trans_y = random.uniform(-0.2,0.2)
        scale=random.uniform(0.7,1.5)
        rotate=random.uniform(-30,30)
        shear=random.uniform(-10,10)
    # shear = 1
        aug_affine = iaa.Affine(scale=scale,rotate=rotate, shear=shear,translate_percent={"x": trans_x, "y": trans_y})
        aug_affine_lbl = iaa.Affine(scale=scale,rotate=rotate, shear=shear,translate_percent={"x": trans_x, "y": trans_y}, order=0, cval=0)
    else:
        trans_x = random.randint(-3,4)
        trans_y = random.randint(-3,4)
    
        aug_affine = iaa.Affine(translate_px={"x": trans_x, "y": trans_y}) 
        aug_affine_lbl = iaa.Affine(translate_px={"x": trans_x, "y": trans_y}, order=0, cval=0) 
    
    mask = np.ones((label.shape), dtype=np.int32)
    
    unique_labels = np.unique(label)
    valid_pts, _ = cv2.projectPoints(id2centers[unique_labels], rvec, tvec, np.array(k_matrix), None)
    valid_pts = np.squeeze(valid_pts, axis=1)
    keypts = imgaug.augmentables.kps.KeypointsOnImage.from_xy_array(valid_pts, label.shape)

    aug_img = aug_affine.augment_image(img)
    aug_mask = aug_affine.augment_image(mask)
    aug_label = aug_affine_lbl.augment_image(label)
    
    affined_keypts = aug_affine.augment_keypoints(keypts)
    aug_valid_pts = affined_keypts.to_xy_array()
    aug_overall_pts = np.zeros((id2centers.shape[0], 2), dtype=np.float32)
    aug_overall_pts[unique_labels] = aug_valid_pts
    
    aug_vertex2d = generate_vote_from_pts(aug_label, aug_overall_pts, aug_mask)
    
    return aug_img, aug_label, aug_vertex2d, aug_mask.astype(np.float32)

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask):
        for t in self.transforms:
            img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask = t(img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask)
        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        seg_target = torch.from_numpy(seg_target).float()
        if mask is not None:
            mask = torch.from_numpy(mask)
        vertex_target = torch.from_numpy(vertex_target).float().permute(2, 0, 1)
        ori_img = np.array(ori_img).astype(np.float32)
        ori_img = torch.from_numpy(ori_img).float()
        
        pose_target = torch.from_numpy(pose_target).float()
        camera_k_matrix = torch.from_numpy(camera_k_matrix).float()

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask

class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        
    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask):
        img = self.color_jitter(Image.fromarray(img))
        img = np.array(img)

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img, mask

class RandomHorizontalFlip(object):
    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            ori_img = cv2.flip(ori_img, 1)
            seg_target = cv2.flip(seg_target, 1)
            vertex_target = cv2.flip(vertex_target, 1)
            

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img
    
class RandomVerticalFlip(object):
    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img):
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            ori_img = cv2.flip(ori_img, 0)
            seg_target = cv2.flip(seg_target, 0)
            vertex_target = cv2.flip(vertex_target, 0)

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img):
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        ori_img = ori_img.rotate(rotate_degree, Image.BILINEAR)
        seg_target = seg_target.rotate(rotate_degree, Image.NEAREST)
        vertex_target = vertex_target.rotate(rotate_degree, Image.NEAREST)

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img


class RandomGaussianBlur(object):
    def __init__(self, blur_radius_list):
        self.blur_radius_list = blur_radius_list
    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img):
        if random.random() < 0.5:
            blur_radius = np.random.choice(self.blur_radius_list)
            img, ori_img = Image.fromarray(img), Image.fromarray(ori_img)
            img = img.filter(ImageFilter.GaussianBlur(
                radius=blur_radius))
            ori_img = ori_img.filter(ImageFilter.GaussianBlur(
                radius=blur_radius))
            img, ori_img = np.array(img), np.array(ori_img)

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img


class RandomScaleCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img):
        if self.padding > 0:
            p = self.padding
            img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value = 0)
            ori_img = cv2.copyMakeBorder(ori_img, p, p, p, p, cv2.BORDER_CONSTANT, value = 0)
            seg_target = cv2.copyMakeBorder(seg_target, p, p, p, p, cv2.BORDER_CONSTANT, value = 0)
            vertex_target = cv2.copyMakeBorder(vertex_target, p, p, p, p, cv2.BORDER_CONSTANT, value = 0)
        
        assert img.shape[:2] == seg_target.shape[:2]
        h, w = img.shape[:2]
        th, tw = self.size
        # print(th, tw, h, w)
        if w == tw and h == th:
            return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img
        if w < tw or h < th:
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            ori_img = cv2.resize(ori_img, (tw, th), interpolation=cv2.INTER_NEAREST)
            seg_target = cv2.resize(seg_target, (tw, th), interpolation=cv2.INTER_NEAREST)
            vertex_target = cv2.resize(vertex_target, (tw, th), interpolation=cv2.INTER_NEAREST)
            return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img

        cx = (w - tw) // 2
        cy = (h - th) // 2
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img[y1:y1 + th, x1: x1 + tw, :]
        ori_img = ori_img[y1:y1 + th, x1: x1 + tw, :]
        seg_target = seg_target[y1:y1 + th, x1: x1 + tw]
        vertex_target = vertex_target[y1:y1 + th, x1: x1 + tw, :]
        camera_k_matrix[0, 2] = camera_k_matrix[0, 2] - (cx - x1)
        camera_k_matrix[1, 2] = camera_k_matrix[1, 2] - (cy - y1)
        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img
    
    
class CenterCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img):
        if self.padding > 0:
            p = self.padding
            img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value = 0)
            ori_img = cv2.copyMakeBorder(ori_img, p, p, p, p, cv2.BORDER_CONSTANT, value = 0)
            seg_target = cv2.copyMakeBorder(seg_target, p, p, p, p, cv2.BORDER_CONSTANT, value = 0)
            vertex_target = cv2.copyMakeBorder(vertex_target, p, p, p, p, cv2.BORDER_CONSTANT, value = 0)


        assert img.shape[:2] == seg_target.shape[:2]
        w, h = img.shape[:2]
        th, tw = self.size
        if w == tw and h == th:
            return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img
        if w < tw or h < th:
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            ori_img = cv2.resize(ori_img, (tw, th), interpolation=cv2.INTER_NEAREST)
            seg_target = cv2.resize(seg_target, (tw, th), interpolation=cv2.INTER_NEAREST)
            vertex_target = cv2.resize(vertex_target, (tw, th), interpolation=cv2.INTER_NEAREST)
            return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img

        cx = (w - tw) // 2
        cy = (h - th) // 2
        img = img[cx: cx + tw, cy:cy + th, :]
        ori_img = ori_img[cx: cx + tw, cy:cy + th, :]
        seg_target = seg_target[cx: cx + tw, cy:cy + th]
        vertex_target = vertex_target[cx: cx + tw, cy:cy + th, :]
        
        camera_k_matrix[0, 2] = camera_k_matrix[0, 2] - cx
        camera_k_matrix[1, 2] = camera_k_matrix[1, 2] - cy
        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img):
        assert img.size == seg_target.size

        img = img.resize(self.size, Image.BILINEAR)
        seg_target = seg_target.resize(self.size, Image.NEAREST)
        vertex_target = vertex_target.resize(self.size, Image.NEAREST)

        return img, seg_target, vertex_target, pose_target, camera_k_matrix, ori_img
