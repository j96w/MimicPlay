from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import json
import cv2
import torch
import shutil, copy
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, \
    vis_detections_filtered_objects_PIL, vis_detections_filtered_objects, calculate_center  # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import h5py
import moviepy.editor as mpy
from moviepy.editor import *
import matplotlib.pyplot as plt

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--left_video',
                        help='left view video',
                        default='vis_1.mp4', type=str)
    parser.add_argument('--right_video',
                        help='right view video',
                        default='vis_2.mp4', type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--hdf5_path',
                        help='h5py file path',
                        default='demo_hand_loc_0.hdf5', type=str)
    parser.add_argument('--target_hdf5_path',
                        help='target h5py file path',
                        default='demo_hand_loc_1.hdf5', type=str)
    parser.add_argument('--hdf5_read_mode',
                        help='online / offline',
                        default='online', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results',
                        default="images_det")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=132028, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True)
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.5,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.5,
                        type=float,
                        required=False)
    parser.add_argument('--downsample_rate', default=1,
                        type=int, help='set to n: sample 1 frame every n frames')

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def export_images_from_hdf5(hdf5_path='', target_path='images', demo_name='demo_1', front_image_index=None):
    shutil.rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)
    h5py_file = h5py.File(hdf5_path, "r+")['data']
    for demo_index in h5py_file.keys():
        if demo_index != demo_name:
            continue
        os.makedirs(os.path.join(target_path), exist_ok=True)
        demo = h5py_file[demo_index]
        obs = demo['obs']
        front_image = obs['front_image_{}'.format(front_image_index)]
        print(front_image.shape)
        image = np.array(front_image).astype("uint8")
        print(image.max(), image.min())
        for img_idx in range(front_image.shape[0]):
            rgb_img = front_image[img_idx]
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(target_path, '{:06d}.png'.format(img_idx)), bgr_img)


def extractImages(pathIn, pathOut=None):
    count = 0
    frame_list = []
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success, image = vidcap.read()
        # print ('Read a new frame: ', success)
        if success:
            frame_list.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
            count = count + 1
    print ('Read {} frames.'.format(len(frame_list)))
    return frame_list


if __name__ == '__main__':

    args = parse_args()

    # print('Called with args:')
    # print(args)

    # hdf5/data/demo_0/obs/[front_image_0, front_image_1, ]
    # demo.keys():  <KeysViewHDF5 ['actions', 'dones', 'interventions', 'next_obs', 'obs', 'policy_acting', 'rewards', 'states', 'user_acting']>
    # obs.keys():  <KeysViewHDF5 ['ee_pose', 'front_image_1', 'front_image_2', 'gripper_position', 'hand_loc', 'joint_positions', 'joint_velocities', 'wrist_image']>

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    new_f_out = h5py.File(args.target_hdf5_path[:-5] + '_new.hdf5', "w")

    demo_name_list = ['demo_{}'.format(i) for i in range(1)]
    val_demo_name = 'demo_1'
    front_image_indexes = [1, 2]
    for front_image_index in front_image_indexes:
        if front_image_index == 1:
            input_video_path = args.left_video
        elif front_image_index == 2:
            input_video_path = args.right_video
        image = extractImages(input_video_path)
        image = np.asarray(image)
        image = image[::args.downsample_rate]
        img_size = 120.
        for demo_name in demo_name_list:
            demo_dir = os.path.join(args.save_dir, demo_name)
            os.makedirs(demo_dir, exist_ok=True)
            demo_name = demo_name
            mpy_frame_list = []
            print('Processing {} front_image_index {} ...'.format(demo_name, front_image_index))
            if args.hdf5_path is not None and args.hdf5_read_mode == 'offline':
                export_images_from_hdf5(args.hdf5_path, demo_name=demo_name, front_image_index=front_image_index)

            with torch.no_grad():
                if args.cuda > 0:
                    cfg.CUDA = True

                if args.cuda > 0:
                    fasterRCNN.cuda()

                fasterRCNN.eval()

                start = time.time()
                max_per_image = 100
                thresh_hand = args.thresh_hand
                thresh_obj = args.thresh_obj
                vis = args.vis

                num_images = len(image)
                num_samples = 0 # num_images
                imglist = ['{:06d}.png'.format(i) for i in range(num_images)]

                print('Loaded Photo: {} images.'.format(num_images))
                hand_det_result = []
                while (num_images > 0):
                    total_tic = time.time()
                    num_images -= 1

                    im_in = image[num_images]
                    im_in = cv2.cvtColor(im_in, cv2.COLOR_RGB2BGR)
                    # bgr
                    im = im_in
                    im = cv2.resize(im, [int(img_size), int(img_size)])
                    mpy_frame_list.append(im)
                    blobs, im_scales = _get_image_blob(im)
                    assert len(im_scales) == 1, "Only single-image batch implemented"
                    im_blob = blobs
                    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

                    im_data_pt = torch.from_numpy(im_blob)
                    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
                    im_info_pt = torch.from_numpy(im_info_np)

                    with torch.no_grad():
                        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                        gt_boxes.resize_(1, 1, 5).zero_()
                        num_boxes.resize_(1).zero_()
                        box_info.resize_(1, 1, 5).zero_()

                        # pdb.set_trace()
                    det_tic = time.time()

                    rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

                    scores = cls_prob.data
                    boxes = rois.data[:, :, 1:5]

                    # extact predicted params
                    contact_vector = loss_list[0][0]  # hand contact state info
                    offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
                    lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

                    # get hand contact
                    _, contact_indices = torch.max(contact_vector, 2)
                    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

                    # get hand side
                    lr = torch.sigmoid(lr_vector) > 0.5
                    lr = lr.squeeze(0).float()

                    if cfg.TEST.BBOX_REG:
                        # Apply bounding-box regression deltas
                        box_deltas = bbox_pred.data
                        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                            # Optionally normalize targets by a precomputed mean and stdev
                            if args.class_agnostic:
                                if args.cuda > 0:
                                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                else:
                                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                                box_deltas = box_deltas.view(1, -1, 4)
                            else:
                                if args.cuda > 0:
                                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                        cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                else:
                                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                    else:
                        # Simply repeat the boxes, once for each class
                        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                    pred_boxes /= im_scales[0]

                    scores = scores.squeeze()
                    pred_boxes = pred_boxes.squeeze()
                    det_toc = time.time()
                    detect_time = det_toc - det_tic
                    misc_tic = time.time()
                    if vis:
                        im2show = np.copy(im)
                    obj_dets, hand_dets = None, None
                    for j in xrange(1, len(pascal_classes)):
                        # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                        if pascal_classes[j] == 'hand':
                            inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
                        elif pascal_classes[j] == 'targetobject':
                            inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

                        # if there is det
                        if inds.numel() > 0:
                            cls_scores = scores[:, j][inds]
                            _, order = torch.sort(cls_scores, 0, True)
                            if args.class_agnostic:
                                cls_boxes = pred_boxes[inds, :]
                            else:
                                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds],
                                                offset_vector.squeeze(0)[inds], lr[inds]), 1)
                            cls_dets = cls_dets[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                            cls_dets = cls_dets[keep.view(-1).long()]
                            if pascal_classes[j] == 'targetobject':
                                obj_dets = cls_dets.cpu().numpy()
                            if pascal_classes[j] == 'hand':
                                hand_dets = cls_dets.cpu().numpy()
                                # print('hand_dets: ', hand_dets.shape)   # (1, 10)
                    if hand_dets is not None:
                        save_hand_dets = hand_dets[0:1]
                        save_hand_dets = np.concatenate((save_hand_dets, np.array(calculate_center(save_hand_dets[0, :4]))[None]), axis=1)
                        save_hand_dets[:, :4] = save_hand_dets[:, :4].astype(np.float32) / img_size
                        save_hand_dets[:, -2:] = save_hand_dets[:, -2:].astype(np.float32) / img_size
                        hand_det_result.append(copy.deepcopy(save_hand_dets))
                    else:
                        save_hand_dets = np.zeros((1, 10))
                        save_hand_dets = np.concatenate((save_hand_dets, np.array(calculate_center(save_hand_dets[0, :4]))[None]), axis=1)
                        save_hand_dets[:, :4] = save_hand_dets[:, :4].astype(np.float32) / img_size
                        save_hand_dets[:, -2:] = save_hand_dets[:, -2:].astype(np.float32) / img_size
                        hand_det_result.append(copy.deepcopy(save_hand_dets))

                    if vis:
                        # visualization
                        # print(num_images, 'save_hand_dets', save_hand_dets)
                        im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

                    misc_toc = time.time()
                    nms_time = misc_toc - misc_tic

                    sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                                    .format(num_images + 1, len(imglist), detect_time, nms_time))
                    sys.stdout.flush()

                    if vis:
                        result_path = os.path.join(demo_dir, imglist[num_images][:-4] + '_{}'.format(front_image_index) + "_det.png")
                        im2show.save(result_path)
                    else:
                        im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
                        cv2.imshow("frame", im2showRGB)
                        total_toc = time.time()
                        total_time = total_toc - total_tic
                        frame_rate = 1 / total_time
                        print('Frame rate:', frame_rate)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                print('Writing HDF5 file to {}'.format(args.target_hdf5_path))
                hand_loc = np.asarray(hand_det_result)[::-1]
                x = copy.deepcopy(hand_loc[:, :, -2])
                y = copy.deepcopy(hand_loc[:, :, -1])
                hand_loc[:, :, -2] = copy.deepcopy(y)
                hand_loc[:, :, -1] = copy.deepcopy(x)
                print('hand_loc.shape: ', hand_loc.shape)  # hand_loc.shape:  (145, 1, 12)
                shifted_hand_loc_1_array = np.concatenate((hand_loc, hand_loc[-1:]), axis=0)
                hand_act = shifted_hand_loc_1_array[1:] - shifted_hand_loc_1_array[:-1]
                print('hand_act.shape: ', hand_act.shape)  # hand_act.shape:  (145, 1, 12)
                obs_path = 'data/'+demo_name+'/obs'
                new_f_out.create_dataset(obs_path+'/hand_loc_{}'.format(front_image_index), data=hand_loc)
                new_f_out.create_dataset(obs_path + '/hand_act_{}'.format(front_image_index), data=hand_act)
                new_f_out.create_dataset(obs_path + '/front_image_{}'.format(front_image_index), data=image)
                if front_image_index == 1:
                    new_f_out.create_dataset(obs_path + '/agentview_image'.format(front_image_index), data=image)
                else:
                    new_f_out.create_dataset(obs_path + '/agentview_image_{}'.format(front_image_index), data=image)
                print('new_f_out[obs_path].keys(): ', new_f_out[obs_path].keys())
                # new_f_out[obs_path].keys():  <KeysViewHDF5 ['front_image_1', 'front_image_2', 'hand_act_1', 'hand_act_2', 'hand_loc_1', 'hand_loc_2']>
                if front_image_index == 1:
                    hand_loc_1 = copy.deepcopy(hand_loc)
                elif front_image_index == 2:
                    all_hand_loc = np.concatenate((hand_loc_1[:, :, -2:], hand_loc[:, :, -2:]), axis=2)
                    new_f_out.create_dataset(obs_path + '/hand_loc', data=all_hand_loc)
                    print('all_hand_loc.shape: ', all_hand_loc.shape)  # hand_act.shape:  (145, 1, 4)
                    all_skip_hand_loc = []
                    num_future_frame = 10
                    skip_len = 2
                    T = all_hand_loc.shape[0]
                    num_samples += T-1
                    for i in range(all_hand_loc.shape[0]):
                        each_skip_hand_loc = []
                        for j in range(num_future_frame):
                            if i + j * skip_len >= all_hand_loc.shape[0]:
                                each_skip_hand_loc.append(all_hand_loc[-1])
                            else:
                                each_skip_hand_loc.append(all_hand_loc[i + j * skip_len])
                        all_skip_hand_loc.append(each_skip_hand_loc)
                    all_skip_hand_loc = np.asarray(all_skip_hand_loc)
                    print('all_skip_hand_loc.shape: ', all_skip_hand_loc.shape)
                    # all_skip_hand_loc.shape:  (145, 10, 1, 4)
                    all_skip_hand_loc = all_skip_hand_loc.reshape(T, -1)
                    new_f_out.create_dataset('data/'+demo_name+'/actions', data=all_skip_hand_loc)

                    new_f_out.create_dataset(obs_path + '/robot0_eef_pos'.format(front_image_index), data=all_hand_loc)
                    new_f_out.create_dataset(obs_path + '/robot0_eef_pos_future_traj'.format(front_image_index), data=all_skip_hand_loc)

                    # 'actions', 'dones', 'interventions', 'next_obs', 'obs', 'policy_acting', 'rewards', 'states', 'user_acting'
                    new_f_out.create_dataset('data/' + demo_name + '/dones', data=np.zeros((T-1)))
                    new_f_out.create_dataset('data/' + demo_name + '/interventions', data=np.zeros((T, 1)))
                    new_f_out.create_dataset('data/' + demo_name + '/policy_acting', data=np.zeros((T)))
                    new_f_out.create_dataset('data/' + demo_name + '/rewards', data=np.zeros((T-1)))
                    new_f_out.create_dataset('data/' + demo_name + '/states', data=np.zeros((0)))
                    new_f_out.create_dataset('data/' + demo_name + '/user_acting', data=np.zeros((T, 1)))
                    new_f_out['data/{}'.format(demo_name)].attrs["num_samples"] = T - 1
                    print(new_f_out['data/{}'.format(demo_name)].attrs["num_samples"])
                    new_f_out.create_dataset('mask/train', data=[demo_name])
                    new_f_out.create_dataset('mask/valid', data=[val_demo_name])
                # validation demo name
                obs_path = 'data/' + val_demo_name + '/obs'
                new_f_out.create_dataset(obs_path + '/hand_loc_{}'.format(front_image_index), data=hand_loc)
                new_f_out.create_dataset(obs_path + '/hand_act_{}'.format(front_image_index), data=hand_act)
                new_f_out.create_dataset(obs_path + '/front_image_{}'.format(front_image_index), data=image)
                if front_image_index == 1:
                    new_f_out.create_dataset(obs_path + '/agentview_image_{}'.format(front_image_index), data=image)
                else:
                    new_f_out.create_dataset(obs_path + '/agentview_image'.format(front_image_index), data=image)
                print('new_f_out[obs_path].keys(): ', new_f_out[obs_path].keys())
                # new_f_out[obs_path].keys():  <KeysViewHDF5 ['front_image_1', 'front_image_2', 'hand_act_1', 'hand_act_2', 'hand_loc_1', 'hand_loc_2']>
                if front_image_index == 1:
                    hand_loc_1 = copy.deepcopy(hand_loc)
                elif front_image_index == 2:
                    all_hand_loc = np.concatenate((hand_loc_1[:, :, -2:], hand_loc[:, :, -2:]), axis=2)
                    new_f_out.create_dataset(obs_path + '/hand_loc', data=all_hand_loc)
                    # print('all_hand_loc.shape: ', all_hand_loc.shape)  # hand_act.shape:  (145, 1, 4)
                    all_skip_hand_loc = []
                    num_future_frame = 10
                    skip_len = 2
                    T = all_hand_loc.shape[0]
                    for i in range(all_hand_loc.shape[0]):
                        each_skip_hand_loc = []
                        for j in range(num_future_frame):
                            if i + j * skip_len >= all_hand_loc.shape[0]:
                                each_skip_hand_loc.append(all_hand_loc[-1])
                            else:
                                each_skip_hand_loc.append(all_hand_loc[i + j * skip_len])
                        all_skip_hand_loc.append(each_skip_hand_loc)
                    all_skip_hand_loc = np.asarray(all_skip_hand_loc)
                    # all_skip_hand_loc.shape:  (145, 10, 1, 4)
                    all_skip_hand_loc = all_skip_hand_loc.reshape(T, -1)
                    new_f_out.create_dataset('data/' + val_demo_name + '/actions', data=all_skip_hand_loc)

                    new_f_out.create_dataset(obs_path + '/robot0_eef_pos'.format(front_image_index), data=all_hand_loc)
                    new_f_out.create_dataset(obs_path + '/robot0_eef_pos_future_traj'.format(front_image_index),
                                             data=all_skip_hand_loc)
                    # 'actions', 'dones', 'interventions', 'next_obs', 'obs', 'policy_acting', 'rewards', 'states', 'user_acting'
                    new_f_out.create_dataset('data/' + val_demo_name + '/dones', data=np.zeros((T - 1)))
                    new_f_out.create_dataset('data/' + val_demo_name + '/interventions', data=np.zeros((T, 1)))
                    new_f_out.create_dataset('data/' + val_demo_name + '/policy_acting', data=np.zeros((T)))
                    new_f_out.create_dataset('data/' + val_demo_name + '/rewards', data=np.zeros((T - 1)))
                    new_f_out.create_dataset('data/' + val_demo_name + '/states', data=np.zeros((0)))
                    new_f_out.create_dataset('data/' + val_demo_name + '/user_acting', data=np.zeros((T, 1)))
                    new_f_out['data/{}'.format(val_demo_name)].attrs["num_samples"] = T - 1
                    print(new_f_out['data/{}'.format(val_demo_name)].attrs["num_samples"])

                data = new_f_out['data']
                data.attrs['total'] = num_samples # T - 1  # num_samples + num_samples
                env_meta = {
                    "env_name": "Libero_Kitchen_Tabletop_Manipulation",
                    "env_version": "1.4.1",
                    "type": 1,
                    "env_kwargs": {
                        "robots": [
                            "Panda"
                        ],
                        "controller_configs": {
                            "type": "OSC_POSE",
                            "input_max": 1,
                            "input_min": -1,
                            "output_max": [
                                0.05,
                                0.05,
                                0.05,
                                0.5,
                                0.5,
                                0.5
                            ],
                            "output_min": [
                                -0.05,
                                -0.05,
                                -0.05,
                                -0.5,
                                -0.5,
                                -0.5
                            ],
                            "kp": 150,
                            "damping_ratio": 1,
                            "impedance_mode": "fixed",
                            "kp_limits": [
                                0,
                                300
                            ],
                            "damping_ratio_limits": [
                                0,
                                10
                            ],
                            "position_limits": None,
                            "orientation_limits": None,
                            "uncouple_pos_ori": True,
                            "control_delta": True,
                            "interpolation": None,
                            "ramp_ratio": 0.2
                        },
                        "bddl_file_name": None,
                        "reward_shaping": False,
                        "camera_names": [
                            "agentview",
                            "robot0_eye_in_hand"
                        ],
                        "camera_heights": 84,
                        "camera_widths": 84,
                        "has_renderer": False,
                        "has_offscreen_renderer": True,
                        "ignore_done": True,
                        "use_object_obs": True,
                        "use_camera_obs": True,
                        "camera_depths": False,
                        "render_gpu_device_id": 0
                    }
                }
                data.attrs['env_args'] = json.dumps(env_meta, indent=4)
                print('Save to {}'.format(args.target_hdf5_path[:-5] + '_new.hdf5'))

    new_f_out.close()