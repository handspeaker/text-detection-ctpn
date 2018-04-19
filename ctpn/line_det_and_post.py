from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
import codecs
import math

sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[2] - box[0]) < 5 or np.linalg.norm(box[3] - box[1]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)


def post_process(boxes, img, img_type, scale):
    nearest_box = []
    nearest_dist = 1.0
    ref_point = [img.shape[1] / 2, img.shape[0] / 2]  # x,y
    for box in boxes:
        box[:8] /= scale
        print(box)
        if img_type == 'owner':  # find the nearest and smallest box
            curr_dist = calc_dist(box, ref_point)
            if len(nearest_box) == 0:
                nearest_box = [box]
                nearest_dist = curr_dist
            else:
                if nearest_dist - curr_dist > 0.05: # choose the nearest one
                    nearest_dist = curr_dist
                    nearest_box = [box]
                else:  # choose the shortest one
                    ne_width = nearest_box[0][2] - nearest_box[0][0]
                    curr_width = box[2] - box[0]
                    if ne_width > curr_width:
                        nearest_dist = curr_dist
                        nearest_box = [box]
        elif img_type == 'name':
            pass
        elif img_type == 'addr':
            pass







def ctpn(sess, net, image_name, result_file, img_type):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    resized_img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # detect single text
    scores, boxes = test_ctpn(sess, net, resized_img)
    # connect text into lines
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    # add post process to filter unrelated boxes
    post_process(boxes,resized_img,img_type,scale)
    # draw boxes and write to file
    draw_boxes(resized_img, image_name, boxes, scale)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='data/demo', help='input image folder')
    parser.add_argument('--result_folder', type=str, default='data/result', help='output result folder')
    parser.add_argument('--result_path', type=str, default='data/result.txt', help='output result list path')
    parser.add_argument('--type', type=str, default='owner', help='input image type: addr,name,owner')
    args, unparsed = parser.parse_known_args()

    if os.path.exists(args.result_folder):
        shutil.rmtree(args.result_folder)
    os.makedirs(args.result_folder)

    cfg_from_file('ctpn/text.yml')
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
    # warm up net
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(args.image_folder, '*.jpeg')) + \
               glob.glob(os.path.join(args.image_folder, '*.jpg'))
    result_file = codecs.open(args.result_path, 'w', 'utf-8')
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name, result_file,args.type)
