import cv2
import numpy as np
import torch as t
import torch
import os
import random

from segment_anything import sam_model_registry
from utils.timer import timeit
from PIL import Image
import cv2 as cv
import argparse

def to_categorical(mask, num_classes, channel='channel_first'):
    """
    convert ground truth mask to categorical
    :param mask: the ground truth mask
    :param num_classes: the number of classes
    :param channel: 'channel_first' or 'channel_last'
    :return: the categorical mask
    """
    if channel != 'channel_first' and channel != 'channel_last':
        assert False, r"channel should be either 'channel_first' or 'channel_last'"
    assert num_classes > 1, "num_classes should be greater than 1"
    unique = np.unique(mask)
    assert len(unique) <= num_classes, "number of unique values should be smaller or equal to the num_classes"
    assert np.max(unique) < num_classes, "maximum value in the mask should be smaller than the num_classes"
    if mask.shape[1] == 1:
        mask = np.squeeze(mask, axis=1)
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    eye = np.eye(num_classes, dtype='uint8')
    output = eye[mask]
    if channel == 'channel_first':
        output = np.moveaxis(output, -1, 1)
    return output





img_dir = ""
mask_dir= ""
mask_dir1= ""
line_dir= ""
line_dir1=""

def img_mask(img_dir,mask_dir,line_dir,line_dir1):

    files = os.listdir(img_dir)
    files.sort(key=lambda x: (x[:-4]))
    imgs = np.empty((len(files), 256, 256,3))  # 生成随机矩阵（不是空矩阵里面有数字）
    masks = np.empty((len(files), 256, 256))
    lines=  np.empty((len(files), 256, 256))
    lines1=  np.empty((len(files), 256, 256))
    coordinatess, valuess = [], []
    boxes = []
    previous_value = np.array(
        [[0, 0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70, 70], [80, 80], [90, 90],
         [100, 100], [110, 110], [120, 120], [130, 130], [140, 140], [150, 150], [160, 160], [170, 170], [180, 180],
         [190, 190], [200, 200], [210, 210], [220, 220], [230, 230], [240, 240], [250, 250]])
    for i in range(len(files)):
        img = cv.imread(img_dir + files[i])
        print(img_dir + files[i])
        img=cv.resize(img,(256,256))
        img=np.array(img)
        mask=np.array(Image.open(mask_dir+files[i]), dtype=int)
        mask1 = np.array(Image.open(mask_dir1 + files[i]), dtype=int)
        line=np.array(Image.open(line_dir+files[i]), dtype=int)
        line1 = np.array(Image.open(line_dir1 + files[i]), dtype=int)

        mask[mask == 0] = 9
        mask[mask == 8] = 0
        mask[mask == 9] = 8
        mask1[mask1 == 0] = 9
        mask1[mask1 == 8] = 0
        mask1[mask1 == 9] = 8

        gt2D = np.array(mask1)
        lines11 = np.array(line1)
        hh = np.where(lines11 == 0.0, 0.0, 255.0)

        x, y = np.where(hh > 5)
        coordinates = np.stack((x, y), axis=1)

        if len(coordinates) == 0:
            print("coordinates is none")
            coordinates = previous_value
        else:
            previous_value = coordinates

        random_points = np.random.choice(coordinates.shape[0], size=20, replace=True)  # 这是在sobel里面的
        selected_coordinates = coordinates[random_points, :]  # 这里的值是反的，也就是说先y在x不是【x，y】

        values = gt2D[selected_coordinates[:, 1], selected_coordinates[:, 0]]  # 但是找值的时候却也是【y,x】而不是【x,y】
        coordinatess.append(selected_coordinates)
        valuess.append(values)


        aaa = np.expand_dims(gt2D, axis=0)
        aaa = np.expand_dims(aaa, axis=3)
        aaa = to_categorical(np.array(aaa), num_classes=9, channel="channel_first")
        a = 8
        for _ in range(8):
            a = random.choice([1, 2, 5, 6, 7])
            if aaa[0, a, :, :].max() != 0:
                break
        if aaa[0, a, :, :].max() == 0:
            a = 8
        aa = aaa[0, a, :, :] * 255
        edges = cv.Canny(aa, 5, 20)
        if edges.max() == 0:
            lingjian = [0, 3, 4, 1, 2, 5, 6, 7, 8]
            for i in lingjian:
                aa = aaa[0, i, :, :] * 255
                edges = cv.Canny(aa, 5, 20)
                if edges.max() != 0:
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

        boxes.append(bboxes)
        #print(np.array(coordinatess).shape,np.array(valuess).shape)


        points_label = (np.array(coordinatess), np.array(valuess))



        imgs[i] = np.array(img)
        masks[i]=np.array(mask)
        lines[i]=np.array(line)
        lines1[i] = np.array(line1)
    boxes = np.array(boxes)
    imgss = np.array(imgs, dtype=np.float32)
    masks=np.array(masks)
    lines=np.array(lines)
    lines1=np.array(lines1)

    imgss=np.transpose(imgss,(0,3,1,2))
    imgss = np.array(imgss, dtype=np.float32)

    return imgss ,masks,lines, lines1,points_label,boxes



@timeit
def evaluate_segmentation(weight_dir='', unet_model=None, bs=1, save=False, model_name='', ifhd=True, ifasd=True):
    print("start to evaluate......")
    checkpoint = t.load(weight_dir)
    try:
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        print('load from dict')
    except:
        unet_model.load_state_dict(checkpoint)
        print('load from single state')
    print("model loaded")

    multimask_output=True
    x_batch,mask,z_batch,z_batch1,points_label,boxes=  img_mask(img_dir,mask_dir,line_dir,line_dir1)

    pred = []
    points, label = points_label
    for i in range(0, len(x_batch), bs):
        index = np.arange(i, min(i + bs, len(x_batch)))
        #   print("3333333", index.shape)
        imgs = x_batch[index]
        lines=z_batch[index]
        lines1=z_batch1[index]


        lines=np.expand_dims(lines,axis=1)
        lines1 = np.expand_dims(lines1, axis=1)
        point_label=(np.array(points)[index],np.array(label)[index])

        boxess=boxes[index]

        TT = unet_model(torch.from_numpy(imgs).float().cuda(), multimask_output, args.img_size, point_label, boxess, lines1)
        oS = TT['low_res_logits']
        lineS = TT['line']

        pred1 = oS.cpu().detach().numpy()
        pred1=pred1[0,:,:,:]
        #print(pred1.shape)
        pred.append(pred1)

    pred = np.argmax(pred, axis=1)
    print(pred.shape)



    for j in range (pred.shape[0]):
        preddd=pred[j,:,:]
        mask1=mask[j,:,:]
        for m in range(0,256):
            for h in range(0,256):
                if preddd[m,h]==0.0:
                    preddd[m,h]=8.0
                elif preddd[m,h]==8.0:
                    preddd[m,h]=0.0
                else:
                    continue
        for e in range(0,256):
            for r in range(0,256):
                if mask1[e,r]==0.0:
                    mask1[e,r]=8.0
                elif mask1[e,r]==8.0:
                    mask1[e,r]=0.0
                else:
                    continue
        # plt.imshow(mask1,cmap="gray")
        # plt.show()
        zhezhao=np.where(mask1==0.0,0.0,1.0)
        chu=np.zeros_like(zhezhao)
        for kk in range(0,256):
            for kkk in range(0,256):
                if preddd[kk,kkk]==mask1[kk,kkk]:
                    chu[kk,kkk]=1.0
                else:
                    continue
        # plt.imshow(chu*255*zhezhao,cmap="gray")
        # plt.show()
        print(j,np.sum(chu*zhezhao),np.sum(zhezhao),(np.sum(chu*zhezhao)/np.sum(zhezhao)))#计算的准确率

        predd=np.expand_dims(preddd,axis=2)
        predfuben=np.concatenate([predd,predd,predd],axis=2)
        mask2=np.expand_dims(mask1,axis=2)
        mask3=np.concatenate([mask2,mask2,mask2],axis=2)
        cv.imwrite("path/" + str(j) + "_pred.png", predd)
        cv.imwrite("path/" + str(j) + "_pred.png", mask2)
        for k in range(0,256):
            for m in range(0,256):
                if preddd[k, m] ==8:
                    predfuben[k, m, 0] = 255.0
                    predfuben[k, m, 1] = 255.0
                    predfuben[k, m, 2] = 255.0
                elif preddd[k, m] ==1:
                    predfuben[k, m, 0] = 255
                    predfuben[k, m, 1] = 0
                    predfuben[k, m, 2] = 255
                elif preddd[k, m] ==2:
                    predfuben[k, m, 0] = 0
                    predfuben[k, m, 1] = 255
                    predfuben[k, m, 2] = 0
                elif preddd[k, m] ==3:
                    predfuben[k, m, 0] = 0
                    predfuben[k, m, 1] = 255
                    predfuben[k, m, 2] = 255
                elif preddd[k, m] ==4:
                    predfuben[k, m, 0] = 0
                    predfuben[k, m, 1] = 34
                    predfuben[k, m, 2] = 0
                elif preddd[k, m] ==5:
                    predfuben[k, m, 0] = 34
                    predfuben[k, m, 1] = 0
                    predfuben[k, m, 2] = 0
                elif preddd[k, m] ==6:
                    predfuben[k, m, 0] = 150
                    predfuben[k, m, 1] =150
                    predfuben[k, m, 2] = 150
                elif preddd[k, m] ==7:
                    predfuben[k, m, 0] = 255
                    predfuben[k, m, 1] = 0
                    predfuben[k, m, 2] = 255
                else:
                    predfuben[k, m, 0] = 0
                    predfuben[k, m, 1] = 0
                    predfuben[k, m, 2] = 255.0
        for p in range(0,256):
            for o in range(0,256):
                if mask2[p, o] ==8:
                    mask3[p, o, 0] = 255.0
                    mask3[p, o, 1] = 255.0
                    mask3[p, o, 2] = 255.0
                elif mask2[p, o] ==1:
                    mask3[p, o, 0] = 255
                    mask3[p, o, 1] = 0
                    mask3[p, o, 2] = 255
                elif mask2[p, o] ==2:
                    mask3[p, o, 0] = 0
                    mask3[p, o, 1] = 255
                    mask3[p, o, 2] = 0
                elif mask2[p, o] ==3:
                    mask3[p, o, 0] = 0
                    mask3[p, o, 1] = 255
                    mask3[p, o, 2] = 255
                elif mask2[p, o] ==4:
                    mask3[p, o, 0] = 0
                    mask3[p, o, 1] = 34
                    mask3[p, o, 2] = 0
                elif mask2[p, o] ==5:
                    mask3[p, o, 0] = 34
                    mask3[p, o, 1] = 0
                    mask3[p, o, 2] = 0
                elif mask2[p, o] ==6:
                    mask3[p, o, 0] = 150
                    mask3[p, o, 1] = 150
                    mask3[p, o, 2] = 150
                elif mask2[p, o] ==7:
                    mask3[p, o, 0] = 255
                    mask3[p, o, 1] = 0
                    mask3[p, o, 2] = 255
                else:
                    mask3[p, o, 0] = 0
                    mask3[p, o, 1] = 0
                    mask3[p, o, 2] = 255.0

        predsave=np.concatenate([predfuben[:,:,2:3],predfuben[:,:,1:2],predfuben[:,:,0:1]],axis=2)

        mask4 = np.concatenate([mask3[:, :, 2:3], mask3[:, :, 1:2], mask3[:, :, 0:1]], axis=2)

        cv.imwrite("path/" + str(j) + "_pred.png", predsave)
        cv.imwrite("path/" + str(j) + "_label.png", mask4)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-save", help='whether to save the evaluation result',default="True", action='store_true')
    parser.add_argument("-model_name", help="the name of the model", type=str, default='')
    parser.add_argument("-weight_dir", help="the path to the weight", type=str, default='train_weights_path')
    parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
    parser.add_argument('--ckpt', type=str, default='path/sam_vit_b_01ec64.pth',help='Pretrained checkpoint')  # sam_vit_h_4b8939.pth   sam_vit_b_01ec64
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument("-he", help="whether to use He initializer", action='store_true')
    parser.add_argument("-cvinit", help="whether to use constant variance initializer", action='store_true')
    parser.add_argument("-multicuda", help="whether to use two cuda gpus", action='store_true')
    args = parser.parse_args()

    seed = 0
    np.random.seed(seed)
    t.manual_seed(seed)
    if args.weight_dir == '':
        model_names ={"unet": "rr.pt", "d1": "pp.pt", "d2": "pp", "d1d2": "pp.pt", "d4": "pp.pt","d2d4": "pp.pt", "d1d2d4": "pp.pt"}
        file_name = model_names[args.model_name]
        weight_dir = '../color_weights/' + file_name
    else:
        weight_dir = args.weight_dir
    toprint = "model: "
    if "d1lr" in weight_dir:
        toprint += "d1"
    pointnet = True if 'd4lr' in weight_dir else False
    extpn = True if 'extpn' in weight_dir else False
    extd1=True if 'extd1' in weight_dir else False
    extd2 = True if 'extd2' in weight_dir else False

    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size, num_classes=args.num_classes, checkpoint=args.ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])  # args.ckpt
    unet_model = sam.cuda()
    if 'aug' in weight_dir:
        toprint += 'aug'
    if 'offmh' in weight_dir:
        toprint += '.offmh'
    if 'gn' in weight_dir:
        toprint += '.gn'
    if 'softmax' in weight_dir:
        toprint += '.softmax'
    if 'etpls' in weight_dir:
        toprint += '.etpls'
    if 'Tetpls' in weight_dir:
        toprint += '.Tetpls'
    if extpn:
        toprint += '.extpn'
    if extd1:
        toprint += '.extd1'
    if extd2:
        toprint += '.extd2'
    if toprint != "":
        print(toprint)
    unet_model.eval()
    evaluate_segmentation(weight_dir=weight_dir, unet_model=unet_model, save=args.save, model_name=args.model_name, ifhd=True, ifasd=True)
