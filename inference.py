# Inspired on https://github.com/duongnv0499/Explain-Deformable-DETR/tree/3f222f514a0bba0d0125063300b85aafc5a6030b

import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T
import requests
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from util import box_ops
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine_single import evaluate, train_one_epoch
from models import build_model
import time
import os

#for video inferencing
import imageio
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--num_ref_frames', default=3, type=int, help='number of reference frames')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int)
    parser.add_argument('--interval1', default=20, type=int)
    parser.add_argument('--interval2', default=60, type=int)

    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='vid_single')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--vid_path', default='./data/vid', type=str)
    parser.add_argument('--coco_pretrain', default=False, action='store_true')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--imgs_dir', type=str, help='input images folder for inference')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


transform = T.Compose([
    T.Resize(600),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

label_names = ['Cracks']
colors = ['yellow']


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    t0 = time.time()
    # img_path = os.path.join(args.imgs_dir, img_file)

    # url = 'data:image/jpeg;base64,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'
    # im = Image.open(requests.get(url, stream=True).raw)
    im = Image.open(r"/content/drive/MyDrive/datadummy/Data/testing.jpg")
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    # img=img.to(device)
    img = img.cuda()
    # print("img = ",img)
    # propagate through the model
    outputs = model(img)
    # print("Outputs = ", outputs)
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    keep = scores[0] > 0.8
    boxes = boxes[0, keep]
    labels = labels[0, keep]

    # and from relative [0, 1] to absolute [0, height] coordinates
    im_h, im_w = im.size
    # print('im_h,im_w',im_h,im_w)
    target_sizes = torch.tensor([[im_w, im_h]])
    target_sizes = target_sizes.cuda()
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    print(time.time() - t0)
    # plot_results
    # source_img = Image.open(requests.get(url, stream=True).raw).convert("RGBA")

    source_img = Image.open(r"/content/drive/MyDrive/datadummy/Data/testing.jpg").convert("RGBA")

    #fnt = ImageFont.truetype("/content/content/Deformable-DETR/font/Aaargh.ttf", 18)
    draw = ImageDraw.Draw(source_img)
    #print ('label' , labels.tolist())
    label_list = labels.tolist()
    # print("Boxes",boxes,boxes.tolist())
    i = 0
    for xmin, ymin, xmax, ymax in boxes[0].tolist():
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=colors[label_list[i] - 1])
        # print('--------')
        # print('i= ', i)
        # print('label is = ', label_list[i]-1)
        # print(label_names[label_list[i]-1])
        if ymin - 18 >= 0:
            ymin = ymin - 18
        draw.text((xmin, ymin), label_names[label_list[i] - 1] + " , scores : " + str(scores[0][0].item()), anchor='md', fill=colors[label_list[i] - 1], spacing = 0.2)
        i += 1

    out_imgName = 'test_result1.png'
    source_img.save(out_imgName, "png")
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    print("Outputs", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)