from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker_vot
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
import vot
import cv2

def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    filename, image, templates_z, scores = siam.build_tracking_graph(final_score_sz, design, env)

    # gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video)
    # pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])
    handle = vot.VOT("rectangle")

    tracker_vot(handle, hp, run, design, final_score_sz, filename, image, templates_z, scores)
    # _, precision, precision_auc, iou = _compile_results(gt, bboxes, evaluation.dist_threshold)
    # print evaluation.video + \
    #         ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision +\
    #         ' -- Precision AUC: ' + "%.2f" % precision_auc + \
    #         ' -- IOU: ' + "%.2f" % iou + \
    #         ' -- Speed: ' + "%.2f" % speed + ' --'
    # print


main()