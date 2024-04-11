
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import kornia
from scipy.special import softmax
dic_loss = kornia.losses.DiceLoss()
import torch
print("torch version: {}".format(torch.__version__))
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime
import tqdm
import numpy as np
import os
from segment_anything import sam_model_registry
from networks.GAN import UncertaintyDiscriminator
from utils.callbacks import ModelCheckPointCallback
from data_generator_sam import ImageProcessor, DataGenerator_PointNet
from utils.utils import soft_to_hard_pred
from utils.metric import metrics2, dice_coef_multilabel
from utils.timer import timeit
from torch.nn.modules.loss import CrossEntropyLoss

def bce2d(input, target):
    n, c, h, w = input.size()
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)

    target_trans = target_t.clone()
    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    ignore_index = (target_t > 1)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0

    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    ignore_index = ignore_index.data.cpu().numpy().astype(bool)

    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num
    weight[ignore_index] = 0
    weight = torch.from_numpy(weight)
    weight = weight.cuda()

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score.cuda() * target.cuda())
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(9)
def get_generators(ids_train, ids_valid, ids_train_lge, ids_valid_lge, batch_size=16, n_samples=2000, crop_size=0, mh=False):
    trainA_generator = DataGenerator_PointNet(df=ids_train, channel="channel_first",phase="train",batch_size=batch_size, source="source", crop_size=crop_size,n_samples=n_samples, match_hist=mh, aug=args.aug,data_dir=args.data_dir)
    validA_generator = DataGenerator_PointNet(df=ids_valid, channel="channel_first", phase="valid",batch_size=batch_size, source="source", crop_size=crop_size, n_samples=-1, match_hist=mh, data_dir=args.data_dir)
    trainB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", phase="train",batch_size=batch_size, source="target", crop_size=crop_size, n_samples=n_samples, aug=args.aug,data_dir=args.data_dir)#ifvert=args.d4 or args.d4aux,
    validB_generator = DataGenerator_PointNet(df=ids_valid_lge, channel="channel_first", phase="valid",  batch_size=batch_size, source="target", crop_size=crop_size, n_samples=-1,data_dir=args.data_dir)
    testB_generator = DataGenerator_PointNet(df=ids_train_lge, channel="channel_first", phase="train",batch_size=batch_size, source="target", crop_size=crop_size, n_samples=-1,data_dir=args.data_dir)

    return iter(trainA_generator), iter(validA_generator), iter(trainB_generator), iter(validB_generator), iter(testB_generator)

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.5):
    low_res_logits = outputs['low_res_logits']

    loss_ce = ce_loss(low_res_logits, torch.from_numpy(low_res_label_batch[:]).long().cuda())
    loss_dice = dice_loss(low_res_logits, torch.from_numpy(low_res_label_batch).cuda(), softmax=True)
    loss =(1 -dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def valid_model_with_one_dataset1(seg_model, data_generator, hd=False,multimask_output=False):
    seg_model.eval()
    dice_list = []
    loss_list = []
    line_loss_list = []
    hd_list = []
    with torch.no_grad():
        for imgV, maskV,lineV,lineVs,points_labelV,idV,boxesV in data_generator:
            #z_batch=z_batch[:,0,:,:]
            prediction = seg_model(torch.from_numpy(imgV).float().cuda(), multimask_output, args.img_size,points_labelV, boxesV,lineVs)
            loss, loss_ce, loss_dice = calc_loss(prediction, np.argmax(maskV, axis=1), ce_loss, dice_loss, args.dice_param)

            pred_line = prediction['line']
            l2=bce2d(pred_line, torch.from_numpy(lineV).float().cuda())
            line_loss_list.append(l2.item())
            loss_list.append((loss+l2).item())
            y_pred = prediction['low_res_logits'].cpu().detach().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y_batch = np.argmax(maskV, axis=1)

            result = metrics2(img_pred=y_pred, img_gt=y_batch, apply_hd=hd, apply_asd=False)
            dice_list.append((result["1"][0] + result["2"][0] + result["3"][0] + result['4'][0]+result["5"][0] + result["6"][0] + result["7"][0] + result['8'][0])/ 8.)
            if hd:

                hd_list.append((result["1"][1] + result["2"][1] + result["3"][1] + result['4'][1] +result["5"][1] + result["6"][1] + result["7"][1] + result['8'][1])/ 8.)
    output = {}
    output["dice"] = np.mean(np.array(dice_list))
    output["loss"] = np.mean(np.array(loss_list))
    output["line_loss"] = np.mean(np.array(line_loss_list))
    if hd:
        output["hd"] = np.mean(np.array(hd_list))
    return output,prediction,idV


def valid_model_with_one_dataset(seg_model, data_generator, hd=False,multimask_output=False):
    seg_model.eval()
    dice_list = []
    loss_list = []
    line_loss_list = []
    hd_list = []
    with torch.no_grad():
        for imgV, maskV,lineV,lineVs,points_labelV,idV,boxesV in data_generator:#, z_batch
            #z_batch=z_batch[:,0,:,:]
            prediction = seg_model(torch.from_numpy(imgV).float().cuda(), multimask_output, args.img_size,points_labelV, boxesV,lineVs)

            out_pred_line = prediction['line'].cpu().detach().numpy()
            out_pred_mask1 = prediction['low_res_logits'].cpu().detach().numpy()
            predT_gailv = softmax(out_pred_mask1, axis=1)
            gailv=np.max(predT_gailv,axis=1)

            #np.savetxt("path/txt/" + idV[0][:-4]+".txt ", gailv[0,:,:], fmt='%.18f', delimiter=',')

            out_pred_mask = np.argmax(out_pred_mask1, axis=1)

            # result = np.full((args.bs,256, 256), 255)
            # result[gailv[:,:] > 0.9] = out_pred_mask[gailv[:,:] > 0.9]
            # cv2.imwrite( "path/txt/" +idV[0], result[0,:,:])
            # np.savetxt("path/txt/" + idV[0][:-4]+".txt ", result[0,:,:], fmt='%.18f', delimiter=',')



            #cv2.imwrite( "path/mask_xunhuan/" + idV[0], out_pred_mask[0, :, :])
            # cv2.imwrite( "path/mask_xunhuan/" + idV[1],out_pred_mask[1, :, :])
            # cv2.imwrite( "path/mask_xunhuan/" + idV[2],out_pred_mask[2, :, :])
            # cv2.imwrite( "path/mask_xunhuan/" + idV[3],out_pred_mask[3, :, :])
            # cv2.imwrite("path/mask_xunhuan/" + idV[4],out_pred_mask[4, :, :])
            # cv2.imwrite("path/mask_xunhuan/" + idV[5], out_pred_mask[5, :, :])
            # cv2.imwrite("path/mask_xunhuan/" + idV[6], out_pred_mask[6, :, :])
            # cv2.imwrite("path/mask_xunhuan/" + idV[7], out_pred_mask[7, :, :])

            #cv2.imwrite( "path/line_xunhuan/" + idV[0],out_pred_line[0, 0, :, :])
            # cv2.imwrite( "path/line_xunhuan/" + idV[1], out_pred_line[1, 0, :, :])
            # cv2.imwrite("path/line_xunhuan/" + idV[2],out_pred_line[2, 0, :, :])
            # cv2.imwrite( "path/line_xunhuan/" + idV[3],out_pred_line[3, 0, :, :])
            # cv2.imwrite( "path/line_xunhuan/" +idV[4], out_pred_line[4, 0, :, :] )
            # cv2.imwrite("path/line_xunhuan/" + idV[5], out_pred_line[5, 0, :, :] )
            # cv2.imwrite("path/line_xunhuan/" + idV[6], out_pred_line[6, 0, :, :] )
            # cv2.imwrite( "path/line_xunhuan/" + idV[7],out_pred_line[7, 0, :, :])

            loss, loss_ce, loss_dice = calc_loss(prediction, np.argmax(maskV, axis=1), ce_loss, dice_loss, args.dice_param)

            pred_line = prediction['line']
            l2=bce2d(pred_line, torch.from_numpy(lineV).float().cuda())
            line_loss_list.append(l2.item())
            loss_list.append((loss+l2).item())
            y_pred = prediction['low_res_logits'].cpu().detach().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y_batch = np.argmax(maskV, axis=1)

            result = metrics2(img_pred=y_pred, img_gt=y_batch, apply_hd=hd, apply_asd=False)
            dice_list.append((result["1"][0] + result["2"][0] + result["3"][0] + result['4'][0]+result["5"][0] + result["6"][0] + result["7"][0] + result['8'][0])/ 8.)
            if hd:
                hd_list.append((result["1"][1] + result["2"][1] + result["3"][1] + result['4'][1] +result["5"][1] + result["6"][1] + result["7"][1] + result['8'][1])/ 8.)
    output = {}
    output["dice"] = np.mean(np.array(dice_list))

    dice=np.reshape(np.mean(np.array(dice_list)),(1,))
    dice = dice.item()
    dice = "{:.20f}".format(dice)
    with open("path/input_vit_sam/"+"ct_dice.txt", "w") as f:
        f.write(dice)

    output["loss"] = np.mean(np.array(loss_list))
    output["line_loss"] = np.mean(np.array(line_loss_list))
    if hd:
        output["hd"] = np.mean(np.array(hd_list))
    return output,prediction,idV


@timeit
def valid_model(seg_model, validA_iterator, validB_iterator):#, testB_generator
    valid_result = {}
    seg_model.eval()
    print("start to valid")
    output,prediction2,idV2 = valid_model_with_one_dataset(seg_model=seg_model, data_generator=validB_iterator, hd=False)
    val_lge_dice = output['dice']
    val_lge_loss = output['loss']
    val_lge_line_loss = output["line_loss"]

    output,prediction1,idV1 = valid_model_with_one_dataset1(seg_model=seg_model, data_generator=validA_iterator, hd=False)
    val_dice = output["dice"]
    val_loss = output['loss']
    val_line_loss=output["line_loss"]

    valid_result["val_dice"] = val_dice
    valid_result['val_loss'] = val_loss
    valid_result['val_line_loss'] = val_line_loss
    valid_result["val_lge_line_loss"] = val_lge_line_loss
    valid_result['val_lge_dice'] = val_lge_dice
    valid_result['val_lge_loss'] = val_lge_loss

    return valid_result
@timeit
def train_epoch(model_gen, model_dis1=None, model_dis2=None,optim_gen=None, optim_dis1=None,optim_dis2=None,trainA_iterator=None, trainB_iterator=None,multimask_output=True):
    source_domain_label = 1
    target_domain_label = 0
    #smooth = 1e-7

    model_gen.train()
    if args.d1:
        model_dis1.train()
    if args.d2:
        model_dis2.train()
    train_result = {}
    running_seg_loss = []
    line_loss=[]
    seg_dice = []
    d1_acc1,d1_acc2,d2_acc1,d2_acc2, = [],[],[],[]

    for (imgA, maskA,lineA,lineAs,points_labelA,idA,boxesA), (imgB, _,lineB,lineBs,points_labelB,idB,boxesB) in zip(trainA_iterator, trainB_iterator):

        optim_gen.zero_grad()
        if args.d1:
            optim_dis1.zero_grad()
        if args.d2:
            optim_dis2.zero_grad()
        if args.d1:
            for param in model_dis1.parameters():
                param.requires_grad = False
        if args.d2:
            for param in model_dis2.parameters():
                param.requires_grad = False
        for param in model_gen.parameters():
            param.requires_grad = True

        SS = model_gen(torch.from_numpy(imgA).float().cuda(), multimask_output, args.img_size, points_labelA, boxesA,lineA)
        oS=SS['low_res_logits']
        lineS=SS['line']
        predS_line = lineS
        predS = F.softmax(oS, dim=1)
        loss_ce = ce_loss(oS, torch.from_numpy(np.argmax(maskA, axis=1)).long().cuda())
        loss_dice = dice_loss(oS, torch.from_numpy(np.argmax(maskA, axis=1)), softmax=True)
        loss_seg3=bce2d(lineS,torch.from_numpy(lineA).float().cuda())
        loss_seg2 = 0
        loss_seg1 = 20*loss_seg3+0.2*loss_ce+0.3*loss_dice
        line_loss.append(loss_seg3.item())
        running_seg_loss.append((loss_seg1 + loss_seg2).item())
        loss_seg1.backward()

        y_pred = soft_to_hard_pred(oS.cpu().detach().numpy(), 1)
        seg_dice.append(dice_coef_multilabel(y_true=maskA, y_pred=y_pred,numLabels=9, channel='channel_first'))

        TT = model_gen(torch.from_numpy(imgB).float().cuda(),multimask_output,args.img_size,points_labelB,boxesB,lineBs)
        oT=TT['low_res_logits']

        lineT=TT['line']
        predT = F.softmax(oT, dim=1) if args.softmax else F.sigmoid(oT)

        predT_line=lineT
        loss_adv_diff = 0
        if args.d1 or args.d2:#or args.d2 or args.d4 or args.d4aux
            loss_adv_output = 0
            if args.d1:
                D_out1 = model_dis1(predT)
                loss_adv_output = args.dr * F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_domain_label).cuda())
            loss_adv_line = 0
            if args.d2:
                D_out2 = model_dis2(predT_line)
                loss_adv_line = args.dr * F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_domain_label).cuda())
            loss_adv_diff += args.w1*loss_adv_output+args.w2 * loss_adv_line

        if loss_adv_diff != 0:
            try:
                loss_adv_diff.backward()
            except:
                print("error!!!!")
                print("value of the loss: {}".format(loss_adv_diff.item()))
                print("exit")
                exit(1)
        optim_gen.step()

        if args.d1 or args.d2:
            if args.d1:
                for param in model_dis1.parameters():
                    param.requires_grad = True

            if args.d2:
                for param in model_dis2.parameters():
                    param.requires_grad = True

                for param in model_gen.parameters():
                    param.requires_grad = False

            # 3. train the discriminators with images from source domain
            if args.d1:
                D_out1 = model_dis1(predS.detach())

                loss_D_same1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_domain_label).cuda())#1
                loss_D_same1.backward()
                D_out1 = torch.sigmoid(D_out1.detach()).cpu().numpy()
                D_out1 = np.where(D_out1 >= .5, 1, 0)
                d1_acc1.append(np.mean(D_out1))

            if args.d2:
                D_out2 = model_dis2(predS_line.detach())

                loss_D_same2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_domain_label).cuda())#0
                loss_D_same2.backward()
                D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
                D_out2 = np.where(D_out2 >= .5, 1, 0)
                d2_acc1.append(np.mean(D_out2))

            #target
            if args.d1:
                D_out1 = model_dis1(predT.detach())
                loss_D_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_( target_domain_label).cuda())
                loss_D_diff1.backward()
                D_out1 = torch.sigmoid(D_out1.detach()).cpu().numpy()
                D_out1 = np.where(D_out1 >= .5, 1, 0)
                d1_acc2.append(1 - np.mean(D_out1))


            if args.d2:
                D_out2 = model_dis2(predT_line.detach())
                loss_D_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_domain_label).cuda())
                loss_D_diff2.backward()
                D_out2 = torch.sigmoid(D_out2.detach()).cpu().numpy()
                D_out2 = np.where(D_out2 >= .5, 1, 0)
                d2_acc2.append(1 - np.mean(D_out2))


            # 5. update parameters
            if args.d1:
                optim_dis1.step()
            if args.d2:
                optim_dis2.step()
    train_result["seg_loss"] = np.mean(np.array(running_seg_loss))
    train_result['seg_dice'] = np.mean(np.array(seg_dice))
    train_result['line_s_loss'] = np.mean(np.array(line_loss))

    if args.d1:
        train_result['dis1_acc1'] = np.mean(np.array(d1_acc1))
        train_result['dis1_acc2'] = np.mean(np.array(d1_acc2))
    if args.d2:
        train_result['dis2_acc1'] = np.mean(np.array(d2_acc1))
        train_result['dis2_acc2'] = np.mean(np.array(d2_acc2))
    return train_result

def print_epoch_result(train_result, valid_result, epoch, max_epochs):
    epoch_len = len(str(max_epochs))
    seg_loss, seg_dice,line_s_loss = train_result["seg_loss"], train_result['seg_dice'],train_result['line_s_loss']
    val_dice, val_loss, val_lge_dice, val_lge_loss,val_lge_line_loss,val_line_loss = valid_result["val_dice"], valid_result['val_loss'], valid_result['val_lge_dice'], valid_result['val_lge_loss'], valid_result["val_lge_line_loss"] ,valid_result['val_line_loss']

    print_msg_line1 = f'valid_loss: {val_loss:.5f} ' + f'valid_lge_loss: {val_lge_loss:.5f} '+ f'val_lge_line_loss: {val_lge_line_loss:.5f} ' + f'val_line_loss: {val_line_loss:.5f}'+ f'train_line_loss: {line_s_loss:.5f}'
    print_msg_line2 = f'valid_dice: {val_dice:.5f} ' + f'valid_lge_dice: {val_lge_dice:.5f} '
    print_msg_line1 = f'train_loss: {seg_loss:.5f} ' + print_msg_line1
    print_msg_line2 = f'train_dice: {seg_dice:.5f} ' + print_msg_line2
    if args.d1:
        dis1_acc1, dis1_acc2 = train_result["dis1_acc1"], train_result['dis1_acc2']
        print_msg_line2 += f'd1_acc1: {dis1_acc1: 5f} ' + f'd1_acc2: {dis1_acc2: 5f} '
    if args.d2:
        dis2_acc1, dis2_acc2 = train_result["dis2_acc1"], train_result['dis2_acc2']
        print_msg_line2 += f'd2_acc1: {dis2_acc1: 5f} ' + f'd2_acc2: {dis2_acc2: 5f} '
    print_msg_line1 = f'[{epoch+ 1:>{epoch_len}}/{max_epochs:>{epoch_len}}] ' + print_msg_line1
    print_msg_line2 = ' ' * (2 * epoch_len + 4) + print_msg_line2
    print(print_msg_line1)
    print(print_msg_line2)
@timeit
def main(batch_size=24, n_samples=2000, n_epochs=200):
    source_train = ImageProcessor.split_data(os.path.join(args.data_dir, "source_train_list.csv"))
    source_valid = ImageProcessor.split_data(os.path.join(args.data_dir, "source_val_list.csv"))
    target_train = ImageProcessor.split_data(os.path.join(args.data_dir, 'target_train_list.csv'))
    target_valid = ImageProcessor.split_data(os.path.join(args.data_dir, 'target_val_list.csv'))
    print("Trainining on {} trainA, {} trainB, validating on {} testA and {} testB samples...!!".format(len(source_train),len( target_train), len(source_valid),len( target_valid)))


    trainA_iterator, validA_iterator, trainB_iterator, validB_iterator, testB_generator = get_generators(source_train, source_valid,target_train,target_valid,batch_size=batch_size, n_samples=n_samples, crop_size=0,mh=args.mh)
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size, num_classes=args.num_classes, checkpoint=args.ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])  # args.ckpt
    model_gen = sam.cuda()
    if args.multicuda:
        model_gen.tomulticuda()
    else:
        model_gen.cuda()
    model_dis1 = None
    model_dis2 = None
    if args.d1:
        model_dis1 = UncertaintyDiscriminator(in_channel=9, heinit=args.he, ext=args.extd1).cuda()
    if args.d2:
        model_dis2 = UncertaintyDiscriminator(in_channel=1, heinit=args.he, ext=args.extd2).cuda()
    if args.sgd:
        optim_gen = torch.optim.SGD(model_gen.parameters(),lr=args.lr,momentum=.95,weight_decay=.0005)
    else:
        optim_gen = torch.optim.Adam(model_gen.parameters(),lr=args.lr, betas=(0.9, 0.99))
    optim_dis1 = None
    if args.d1:
        optim_dis1 = torch.optim.SGD( model_dis1.parameters(), lr=args.d1lr, momentum=args.d1mmt if args.dmmt==0.95 else args.dmmt, weight_decay=.0005)
    optim_dis2 = None
    if args.d2:
        optim_dis2 = torch.optim.SGD( model_dis2.parameters(), lr=args.d2lr, momentum=args.d2mmt if args.dmmt==0.95 else args.dmmt, weight_decay=.0005)
    lr_gen = args.lr
    the_epoch = 0
    best_valid_lge_dice = -1
    best_train_result = {}
    best_valid_result = {}
    # create directory for the weights
    root_directory = 'path/weights/'
    if not os.path.exists(root_directory):
        os.mkdir(root_directory)
    weight_dir = root_directory + 'unet_model_checkpoint_{}.pt'.format(appendix)
    best_weight_dir = root_directory + 'best_unet_model_checkpoint_{}.pt'.format(appendix)
    modelcheckpoint_unet = ModelCheckPointCallback(n_epochs=n_epochs, save_best=True,mode="max",best_model_name=best_weight_dir,save_last_model=True,model_name=weight_dir,entire_model=False)
    if args.d1:
        d1_weight_dir = root_directory + 'out_dis_{}.pt'.format(appendix)
        best_d1_weight_dir = root_directory + 'best_out_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis1 = ModelCheckPointCallback(n_epochs=n_epochs,  mode="max", best_model_name=best_d1_weight_dir,save_last_model=True,model_name=d1_weight_dir,entire_model=False)
    if args.d2:
        d2_weight_dir = root_directory + 'line_dis_{}.pt'.format(appendix)
        best_d2_weight_dir = root_directory + 'best_line_dis_{}.pt'.format(appendix)
        modelcheckpoint_dis2 = ModelCheckPointCallback(n_epochs=n_epochs,  mode="max", best_model_name=best_d2_weight_dir,save_last_model=True,model_name=d2_weight_dir,entire_model=False)
    start_epoch = 0
    print("Training started....!")

    seg_dice, seg_loss ,d2_acc1, d2_acc2= [], [],[],[]
    d1_acc1, d1_acc2 = [], []
    val_dice, val_loss, val_lge_dice, val_lge_loss,val_line_loss,val_lge_line_loss = [], [], [], [],[],[]
    seg_lr, disctor2_lr = [], []


    max_time_elapse_epoch = 0
    for epoch in tqdm.trange(start_epoch, n_epochs, desc='Train', ncols=80):

        iter_start_time = datetime.now()
        train_result = train_epoch(model_gen=model_gen, model_dis1=model_dis1,model_dis2=model_dis2,optim_gen=optim_gen, optim_dis1=optim_dis1, optim_dis2=optim_dis2,trainA_iterator=trainA_iterator, trainB_iterator=trainB_iterator)

        seg_loss.append(train_result["seg_loss"])
        seg_dice.append(train_result['seg_dice'])
        if args.d1:
            d1_acc1.append(train_result['dis1_acc1'])
            d1_acc2.append(train_result['dis1_acc2'])
        if args.d2:
            d2_acc1.append(train_result['dis2_acc1'])
            d2_acc2.append(train_result['dis2_acc2'])
        Loss0 = np.array(train_result["seg_loss"])
        # with open("path/loss_accuracy/3.txt", 'a') as file:
        #     # 写入新内容
        #     file.write(str(Loss0) + '\n')

        valid_result = valid_model(seg_model=model_gen, validA_iterator=validA_iterator,validB_iterator=validB_iterator)
        val_dice.append(valid_result["val_dice"])
        val_loss.append(valid_result['val_loss'])
        val_line_loss.append(valid_result['val_line_loss'])
        print("val_line_loss",np.mean(np.array(val_line_loss)))

        val_lge_dice.append(valid_result['val_lge_dice'])
        val_lge_loss.append(valid_result['val_lge_loss'])
        val_lge_line_loss.append(valid_result['val_lge_line_loss'])
        seg_lr.append(optim_gen.param_groups[0]['lr'])

        print_epoch_result(train_result, valid_result, epoch, n_epochs)

        if best_valid_lge_dice < valid_result["val_lge_dice"]:
            best_valid_lge_dice = valid_result["val_lge_dice"]
            best_train_result = train_result
            best_valid_result = valid_result
            the_epoch = epoch + 1
        monitor_score = valid_result["val_lge_dice"]
        modelcheckpoint_unet.step(monitor=monitor_score, model=model_gen, epoch=epoch + 1, optimizer=optim_gen)
        if args.d1:
            modelcheckpoint_dis1.step(monitor=monitor_score, model=model_dis1, epoch=epoch + 1, optimizer=optim_dis1)
        if args.d2:
            modelcheckpoint_dis2.step(monitor=monitor_score, model=model_dis2, epoch=epoch + 1, optimizer=optim_dis2)
        if args.offdecay:

            if (epoch + 1) % 100 == 0:
                lr_gen = lr_gen * 0.2
                for param_group in optim_gen.param_groups:
                    param_group['lr'] = lr_gen
        print("time elapsed: {} hours".format(np.around((datetime.now() - start_time).seconds / 3600., 1)))
        max_time_elapse_epoch = max(np.around((datetime.now() - iter_start_time).seconds), max_time_elapse_epoch)
        print("Best model on epoch {}: train_dice {}, valid_dice {} , lge_dice {}".format(the_epoch, np.round(best_train_result['seg_dice'], decimals=3,), np.round(best_valid_result['val_dice'], decimals=3),np.round(best_valid_result['val_lge_dice'], decimals=3)  ))#      ,np.round(best_valid_result['val_lge_dice'], decimals=3)                                        , np.round(best_valid_result['test_lge_dice'], decimals=3)))#test_lge_dice {}

    # evaluate the model
    print('Evaluate the model')
    best_model_name_base, ext = os.path.splitext(best_weight_dir)
    root_directory = '{}{}{}{}'.format(best_model_name_base, '.Scr', np.around(modelcheckpoint_unet.best_result, 3), ext)
    print("root directory: {}".format(root_directory))
    from evaluate_mmwhs import evaluate_segmentation
    evaluate_segmentation(weight_dir=root_directory, unet_model=model_gen, save=False, model_name='', ifhd=True, ifasd=True)
    writer = SummaryWriter(comment=appendix + ".Scr{}".format(np.around(best_valid_lge_dice, 3)))
    i = 1
    print("write a training summary")
    for t_loss, t_dice, v_loss, v_dice, lge_loss, lge_dice,  u_lr in zip(seg_loss, seg_dice, val_loss,  val_dice,  val_lge_loss,  val_lge_dice,  seg_lr, ):#etp_loss, etp_loss_T       test_lge_loss,test_lge_dice,, eloss, t_eloss  t_lge_loss, t_lge_dice,
        writer.add_scalars('Loss', {'Training': t_loss,'Validation': v_loss, 'Valid_LGE': lge_loss},  i)
        writer.add_scalars('Dice', {'Training': t_dice, 'Validation': v_dice,'Valid_LGE': lge_dice,}, i)
        writer.add_scalar('Lr/unet', u_lr, i)
        i += 1
    if args.d1:
        i = 1
        for a1, a2 in zip(d1_acc1, d1_acc2):
            writer.add_scalar('d1_acc1', a1, i)
            writer.add_scalar('d1_acc2', a2, i)
            i += 1
    if args.d2:
        i = 1
        for a1, a2 in zip(d2_acc1, d2_acc2):
            writer.add_scalar('d2_acc1', a1, i)
            writer.add_scalar('d2_acc2', a2, i)
            i += 1
    writer.close()

def get_appendix():
    appendix = args.apdx + '.lr{}'.format(args.lr_fix)
    if args.nf != 32:
        appendix += '.nf{}'.format(args.nf)
    if args.mmt != 0.95:
        appendix += '.mmt{}'.format(args.mmt)
    if args.dmmt != 0.95:
        appendix += '.dmmt{}'.format(args.dmmt)
    else:
        if args.d1mmt != 0.95:
            appendix += '.d1mmt{}'.format(args.d1mmt)
    if args.d1:
        appendix += '.d1lr{}'.format(args.d1lr)
    if args.w1 != 1:
        appendix += '.w1_{}'.format(args.w1)
    if args.sgd:
        appendix += '.sgd'
    if not args.mh:
        appendix += '.mh'
    if args.aug == 'heavy':
        appendix += '.hvyaug'
    elif args.aug == 'light':
        appendix += '.litaug'
    if args.softmax:
        appendix += '.softmax'
    if not args.offdecay:
        appendix += '.offdecay'
    if args.he:
        appendix += '.he'
    if args.cvinit:
        appendix += '.cv'
    if args.extd1:
        appendix += '.extd1'
    if args.ft:
        appendix += '.ft'

    if args.dr != 0.01:
        appendix += '.dr{}'.format(args.dr)
    return appendix

if __name__ == '__main__':
    start_time = datetime.now()
    seed =- 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    import argparse

    parser = argparse.ArgumentParser()
    # general settings:
    parser.add_argument("-bs", help="the batch size of training", type=int, default=2)
    parser.add_argument("-ns", help="number of samples per epoch", type=int, default=10)
    parser.add_argument("-e", help="number of epochs to train", type=int, default=400)
    parser.add_argument("-offdecay", help="whether not to use learning rate decay for unet", default=True,action='store_false')
    parser.add_argument("-apdx", help="the append ix to the checkpoint", type=str, default='train_point_tf')
    parser.add_argument("-he", help="whether to use He initializer", action='store_true')
    parser.add_argument("-cvinit", help="whether to use constant variance initializer", action='store_true')
    parser.add_argument("-multicuda", help="whether to use two cuda gpus", action='store_true')
    parser.add_argument("-data_dir", help="the directory to the data", type=str, default="data_path/")
    # data augmentation:
    parser.add_argument("-aug", help='the type of the augmentation, should be one of "", "heavy" or "light"', type=str, default="heavy")
    parser.add_argument("-mh", help='turn on "histogram matching" (not needed)', action='store_true')
    # unet setting:
    parser.add_argument("-lr", help="the actual learning rate of the unet", type=float, default=1e-3)
    parser.add_argument("-lr_fix", help="the base learning rate of the unet(used to be written in the 'appendix' string)", type=float, default=1e-3)
    parser.add_argument("-sgd", help="whether to use sgd for the unet",action='store_true')
    parser.add_argument("-nf", help="base number of the filters for the unet", type=int, default=32)
    parser.add_argument("-out_ch", help="the out channels of the first conv layer(only used for SKUnet) (deprecated)", type=int,  default=32)
    parser.add_argument("-drop", help="whether to apply dropout in the decoder of the unet", action='store_true')
    parser.add_argument("-softmax", help="whether to apply softmax as the last activation layer of segmentation model",default=True, action='store_true')
    parser.add_argument("-dice", help="whether to use dice loss instead of jaccard loss", action='store_true')
    parser.add_argument("-mmt", help="the value of momentum when using sgd", type=float, default=0.95)
    # discriminator setting:
    parser.add_argument("-d1", help="whether to apply outer space discriminator", default=True,action='store_true')
    parser.add_argument("-d2", help="whether to apply lines discriminator", default=True, action='store_true')
    parser.add_argument("-d1lr", help="the learning rate for outer space discriminator", type=float, default=2.5e-5)
    parser.add_argument("-d2lr", help="the learning rate for outer space discriminator", type=float, default=2.5e-5)
    parser.add_argument("-ft", help="whether to apply feature transformation layer in pointcloudcls", action='store_true')
    parser.add_argument("-dmmt", help="to set the momentum of the optimizers of the discriminators", type=float, default=0.95)
    parser.add_argument("-d1mmt", help="to set the momentum of the optimizers of the discriminators", type=float, default=0.95)
    parser.add_argument("-d2mmt", help="to set the momentum of the optimizers of the discriminators", type=float,default=0.95)
    parser.add_argument("-extd1", help="whether to extend output discriminator", default=True,action='store_true')
    parser.add_argument("-extd2", help="whether to extend output discriminator", default=True, action='store_true')
    # weights of losses:
    parser.add_argument("-dr", help="the ratio of the adversarial loss for the unet", type=float, default=.01)
    parser.add_argument("-w1", help="the weight for the adversarial loss of unet to the output space discriminator", type=float, default=1.)
    parser.add_argument("-w2", help="the weight for the adversarial loss of unet to the output space discriminator", type=float, default=1.)
    #网络相关参数
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--ckpt', type=str, default='path/checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')  # sam_vit_h_4b8939.pth   sam_vit_b_01ec64
    parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
    parser.add_argument('--dice_param', type=float, default=0.5)
    parser.add_argument('--warmup', default=False, action='store_true',help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')



    args = parser.parse_args()
    assert args.aug == '' or args.aug == 'heavy' or args.aug == 'light'
    appendix = get_appendix()
    print(appendix)
    torch.autograd.set_detect_anomaly(True)
    main(batch_size=args.bs, n_samples=args.ns, n_epochs=args.e)