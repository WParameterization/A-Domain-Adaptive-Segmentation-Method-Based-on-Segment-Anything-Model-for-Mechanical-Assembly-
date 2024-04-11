import numpy as np
import os
from PIL import Image
import cv2 as cv
import random
try:
    np.random.bit_generator = np.random._bit_generator
    print("rename numpy.random._bit_generator")
except:
    print("numpy.random.bit_generator exists")
import pandas as pd

import imgaug as ia
import imgaug.augmenters as iaa
from matplotlib import  pyplot as plt
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from utils.utils import to_categorical

from imgaug.augmenters.meta import Sequential
def augmentation(image, mask=None,line=None,line1=None,mask1=None):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            sometimes(iaa.CropAndPad(percent=(-0.05, 0.1),pad_mode=ia.ALL, pad_cval=(0, 255) )),
            sometimes(iaa.Affine( scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, rotate=(-45, 45),shear=(-16, 16),order=[0, 1],cval=(0, 255),mode=ia.ALL)),
            iaa.SomeOf((0, 5),
                       [ sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           iaa.OneOf([ iaa.GaussianBlur((0, 3.0)),iaa.AverageBlur(k=(2, 7)), iaa.MedianBlur(k=(3, 11)),]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                           iaa.SimplexNoiseAlpha(iaa.OneOf([iaa.EdgeDetect(alpha=(0.5, 1.0)),iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)), ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           iaa.OneOf([iaa.Dropout((0.01, 0.1), per_channel=0.5), iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2), ]),
                           iaa.Invert(0.05, per_channel=True),
                           iaa.Add((-10, 10), per_channel=0.5),
                           iaa.AddToHueAndSaturation((-20, 20)),
                           iaa.OneOf([iaa.Multiply((0.5, 1.5), per_channel=0.5),]),
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    if mask is None and line is None:
        image_heavy = seq(images=image)
        return image_heavy
    else:
        if image.ndim == 4:
            mask = np.array(mask)
            line=np.array(line)
            aa=np.concatenate([mask,line],axis=3)#[image,mask.astype(np.int32), line.astype(np.int32)]
            aa=np.concatenate([aa,line1],axis=3)
            aa=np.concatenate([aa,mask1],axis=3)
            image_heavy ,label_heavy= seq(images=image,segmentation_maps=aa.astype(np.int32))#,mask.astype(np.int32) segmentation_maps=aa(mask.astype(np.int32),line.astype(np.int32)),mask_heavy
            mask_heavy=label_heavy[:,:,:,0:1]
            line_heavy=label_heavy[:,:,:,1:2]
            line1_heavy=label_heavy[:,:,:,2:3]
            mask1_heavy = label_heavy[:, :, :, 3:4]

        return image_heavy,mask_heavy,line_heavy,line1_heavy,mask1_heavy

def sobel(img, threshold):
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i + 3, j:j + 3]))
            h = sum(sum(G_y * img[i:i + 3, j:j + 3]))
            mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))
    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    return mag





def light_aug(images, masks=None, segmap=False):

    sometimes = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.2),
            iaa.Flipud(0.2),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                shear=(-12, 12),
                order=[0, 1],
                cval=(0, 255),
                mode='constant',
            )),
        ],
        random_order=True
    )
    if masks is None:
        image_light = seq(images=images)
        return image_light
    else:
        if segmap:
            segmaps = []
            for mask in masks:
                segmaps.append(SegmentationMapsOnImage(mask.astype(np.int32), shape=images.shape[-3:]))
        else:
            segmaps = np.array(masks, dtype=np.int32)
        image_light,masks = seq(images=images, segmentation_maps=segmaps)
        if segmap:
            mask_light = []
            for mask in masks:
                mask_light.append(mask.get_arr())
            masks = np.array(mask_light)
        from matplotlib import pyplot as plt
        plt.imshow(image_light[0,:,:,0])
        plt.show()
        return image_light, masks


class ImageProcessor:

    @staticmethod
    def split_data(img_path):
        df_train = pd.read_csv(img_path)
        ids_train = df_train['img']
        return ids_train

    @staticmethod
    def crop_volume(vol, crop_size=112):
        return np.array(vol[:,
                        int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                        int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


class DataGenerator_PointNet:
    def __init__(self, df, channel="channel_first", phase="train",  aug='',  batch_size=8,  source="source",  crop_size=0,  n_samples=-1, toprint=False, match_hist=False,  ifvert=False,segmap=False,data_dir=''):

        assert phase == "train" or phase == "valid" or phase == "train1", r"phase has to be either'train' or 'valid'"
        assert source == "source" or source == "target"
        assert aug == '' or aug == 'heavy' or aug == 'light'
        self._data = df
        self._len = len(df)
        self._shuffle_indices = np.arange(len(df))
        self._shuffle_indices = np.random.permutation(self._shuffle_indices)#np.random.permutation
        self._shuffle_indices1 = np.arange(600)
        self._shuffle_indices1 = np.random.permutation(self._shuffle_indices)  # np.random.permutation
        self._source = source
        self._aug = aug
        self._crop_size = crop_size
        self._phase = phase
        self._channel = channel
        self._batch_size = batch_size
        self._index = 0
        self._totalcount = 0
        if n_samples == -1:
            self._n_samples = len(df)
        else:
            self._n_samples = n_samples
        self._toprint = toprint

        self._match_hist = match_hist
        self._vert = ifvert
        self._segmap = segmap
        self._data_dir = data_dir

    def __len__(self):
        return self._len

    def get_image_paths(self, id,id1):
        if self._source == "source":
            if self._phase == "train":
                img_path = os.path.join(self._data_dir, 'path/img/{}'.format(id))
                mask_path_ori = os.path.join(self._data_dir, 'path/mask/{}'.format(id))
                line_path_ori = os.path.join(self._data_dir, 'path/line/{}'.format(id))
                line_path_1 = os.path.join(self._data_dir, 'path/line/{}'.format(id))
                mask_path_1 = os.path.join(self._data_dir, 'path/mask/{}'.format(id))

            else:
                img_path = os.path.join(self._data_dir,  'path/img/{}'.format(id))
                mask_path_ori = os.path.join(self._data_dir, 'path/mask/{}'.format(id))
                line_path_ori = os.path.join(self._data_dir, 'path/line/{}'.format(id))
                line_path_1 = os.path.join(self._data_dir, 'path/line/{}'.format(id))
                mask_path_1 = os.path.join(self._data_dir, 'path/mask/{}'.format(id))
        else:#target
            if self._phase == "train":
                img_path = os.path.join(self._data_dir, 'path/img/{}'.format(id1))
                mask_path_ori = os.path.join(self._data_dir, 'path/mask/{}'.format(id1))
                line_path_ori = os.path.join(self._data_dir, 'path/line_ori/{}'.format(id1))
                line_path_1 = os.path.join(self._data_dir, 'path/line_transunet/{}'.format(id1))
                mask_path_1 = os.path.join(self._data_dir, 'path/mask_transunet/{}'.format(id1))
            else:
                img_path = os.path.join(self._data_dir, 'path/img/{}'.format(id1))
                mask_path_ori = os.path.join(self._data_dir, 'path/mask/{}'.format(id1))
                line_path_ori = os.path.join(self._data_dir, 'path/line_ori/{}'.format(id1))
                line_path_1 = os.path.join(self._data_dir, 'path/line_transunet/{}'.format(id1))
                mask_path_1 = os.path.join(self._data_dir, 'path/mask_transunet/{}'.format(id1))

        return img_path, mask_path_ori,line_path_ori,line_path_1,mask_path_1,id

    def get_images_masks(self, img_path, mask_path, line_path,line_path_1,mask_path_1):
         img, mask,line,line1 = cv.imread(img_path), Image.open(mask_path),Image.open(line_path),Image.open(line_path_1)
         mask1=Image.open(mask_path_1)

         img = cv.resize(img, (256, 256))

         mask1 = mask1.resize((256, 256), Image.NEAREST)
         mask1 = np.array(mask1)
         mask1 = mask1.reshape(256, 256, 1)

         mask = mask.resize((256, 256), Image.NEAREST)
         mask = np.array(mask)
         mask = mask.reshape(256, 256, 1)


         line = line.resize((256, 256), Image.NEAREST)
         line = np.array(line)
         line = line.reshape(256, 256, 1)

         line1 = line1.resize((256, 256), Image.NEAREST)
         line1 = np.array(line1)
         line1 = line1.reshape(256, 256, 1)


         img=np.array(img)
         mask=np.array(mask)
         line=np.array(line)

         mask1=np.array(mask1)
         line1 = np.array(line1)


         mask[mask == 0] = 9
         mask[mask == 8] = 0
         mask[mask == 9] = 8
         mask1[mask1 == 0] = 9
         mask1[mask1 == 8] = 0
         mask1[mask1 == 9] = 8

         return img, mask,line,line1,mask1

    def __iter__(self):
        self._totalcount = 0
        return self

    def __next__(self):
        images, masks,lines,lines1 = [],[],[],[]
        masks1=[]
        path=[]

        indices = []
        if self._totalcount >= self._n_samples:
            self._totalcount = 0
            raise StopIteration
        for i in range(self._batch_size):
            indices.append(self._index)
            self._index += 1
            self._totalcount += 1
            self._index = self._index % self._len
            if self._totalcount >= self._n_samples:
                break
        ids_train_batch = self._data.iloc[self._shuffle_indices[indices]]

        for _id in ids_train_batch.values:
            id1=_id
            img_path, mask_path,line_path,line_path_1,mask_path_1,id1 = self.get_image_paths(id=_id,id1=id1)
            #print(mask_path_1)
            a=np.loadtxt("path/dice.txt")
            if a>0.80:
                if "line_transunet" in line_path_1 and "mask_transunet" in mask_path_1:

                    line_path_1 = line_path_1.replace("line_transunet", "line_xunhuan")
                    mask_path_1 =mask_path_1.replace("mask_transunet", "mask_xunhuan")
                else:
                    line_path_1=line_path_1
                    mask_path_1=mask_path_1
            img, mask,line,line1,mask1= self.get_images_masks(img_path=img_path, mask_path=mask_path,line_path=line_path,line_path_1=line_path_1,mask_path_1=mask_path_1)#, vertex_path=vertex_path
            assert mask.ndim == 3
            assert line.ndim == 3
            assert line1.ndim == 3
            path.append(id1)
            images.append(img)
            masks.append(mask)
            masks1.append(mask1)

            lines.append(line)
            lines1.append(line1)
        images = np.array(images)
        lines = np.array(lines)
        lines1 = np.array(lines1)
        if self._phase=="train" or self._phase=="valid":
            if self._aug == 'heavy' or self._aug == 'light':
                img_min = images.min()
                img_max = images.max()
                images = (images - img_min) * 255. / (img_max - img_min)
                images = np.array(images, dtype=np.uint8)
                images, masks,lines,lines1,masks1 = augmentation(images, masks,lines,lines1,masks1)
                images = img_min + images.astype(np.float32) * (img_max - img_min) / 255.#0-1
                masks = np.array(masks)
                lines = np.array(lines)
                masks1=np.array(masks1)
                lines1=np.array(lines1)

        previous_value = np.array(
            [[0, 0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70, 70], [80, 80], [90, 90],
             [100, 100], [110, 110], [120, 120], [130, 130], [140, 140], [150, 150], [160, 160], [170, 170], [180, 180],
             [190, 190], [200, 200], [210, 210], [220, 220], [230, 230], [240, 240], [250, 250]])
        coordinatess, valuess = [], []
        boxes = []
        for p in range(len(ids_train_batch.values)):
             gt2D = np.array(masks1)[p, :, :, 0]

             lines11=np.array(lines1)[p, :, :, 0]
             hh = np.where(lines11 == 0.0, 0.0, 255.0)

             x, y = np.where(hh > 5)
             coordinates = np.stack((x, y), axis=1)
             if len(coordinates)==0:
                 print("coordinates is none")
                 coordinates=previous_value
             else:
                 previous_value = coordinates

             random_points = np.random.choice(coordinates.shape[0], size=20, replace=True)
             selected_coordinates = coordinates[random_points, :]

             values = gt2D[selected_coordinates[:, 1], selected_coordinates[:, 0]]
             coordinatess.append(selected_coordinates)
             valuess.append(values)
             points_label = (np.array(coordinatess), np.array(valuess))


             aaa = np.expand_dims(gt2D, axis=0)
             aaa = np.expand_dims(aaa, axis=3)
             aaa = to_categorical(np.array(aaa), num_classes=9, channel=self._channel)
             a = 8

             for _ in range(8):
                 a = random.choice([1, 2, 5, 6, 7])
                 if aaa[0, a, :, :].max() != 0:
                     break
             if aaa[0, a, :, :].max() == 0:
                 a = 8
             aa = aaa[0, a, :, :] * 255
             edges = cv.Canny(aa, 5, 20)
             if edges.max()==0:
                lingjian = [0,3,4,1, 2, 5, 6, 7,8]
                for i in lingjian:

                    aa = aaa[0, i, :, :] * 255
                    edges = cv.Canny(aa, 5, 20)
                    if edges.max()!=0:
                        break
                    else:

                        continue

             contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
             largest_contour = max(contours, key=cv.contourArea)
             rect = cv.minAreaRect(largest_contour)
             box = cv.boxPoints(rect)
             bboxes = np.int0(box)

             zhanshidian = np.array(bboxes)
             cv.polylines(aa, [zhanshidian], isClosed=True, color=150, thickness=2)
             # pixel_values = gt2D1.flatten().tolist()
             # unique_value = list(set(pixel_values))
             # sorted_values = sorted(unique_value)
             # valid_values = [x for x in sorted_values if x not in [0, 3, 4, 8]]
             # if len(valid_values) == 0:
             #     select_value = 0
             # else:
             #     select_value = random.choice(valid_values)
             # y_indices, x_indices = np.where(gt2D1 == select_value)
             # x_min, x_max = np.min(x_indices), np.max(x_indices)
             # y_min, y_max = np.min(y_indices), np.max(y_indices)
             #
             # bboxes = np.array([x_min, y_min, x_max, y_max])
             boxes.append(bboxes)

        boxes=np.array(boxes)
        if self._channel == "channel_first":

            images = np.moveaxis(images, -1, 1)
            lines=np.moveaxis(lines,-1,1)
            lines1=np.moveaxis(lines1,-1,1)
        plt.imshow(masks[0,:,:,0])
        plt.show()
        masks = to_categorical(np.array(masks), num_classes=9, channel=self._channel)


        return images, masks,lines,lines1,points_label,path,boxes


if __name__ == "__main__":
    pass
