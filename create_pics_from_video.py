# %matplotlib inline
import math
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints, BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from modules.load_state import load_state
from mrcnn import utils
from val import normalize, pad_width
import skimage.io
import random

# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "rcnn/samples/coco/"))  # To find local version
sys.path.insert(0, "rcnn/samples/coco/")
import samples.coco.coco as coco

nPoints = 18
rad_270 = 4.71239
rad_90 = 1.5708
rad_180 = 3.14
threshold = 0.1

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho',
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip',
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

bones_to_detect = [6, 7, 9, 10]  # 6, 9 - chest; 7, 10 - legs

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

time_period_sleep = 500  # seconds
time_period_wake = 10
time_period = time_period_sleep + time_period_wake
len_points = len(BODY_PARTS_PAF_IDS) - 2
time_anomaly = 0
MAX_PERSONS = 5
max_num_persons = 0
checkpoint_path = 'weights/checkpoint_iter_370000.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
nPoints = 18
rad_270 = 4.71239
rad_90 = 1.5708
SEC_WITHOUT_HELP = 3
anomaly_checker = np.zeros(SEC_WITHOUT_HELP * 3)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


MODEL_DIR = os.path.join('weights', "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('weights', "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print('load is completed')


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
    height, width = img.shape[0], img.shape[1]
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def convert_video_to_pics(video_file, output_folder):
    print('start converting video to pics')
    vidcap = cv2.VideoCapture(video_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(output_folder + "/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    print('done converting video to pics')


def convert_to_gray(in_, out_):
    print('start  convert to gray')
    files = os.listdir(in_)
    if not os.path.exists(out_):
        os.makedirs(out_)
    for name in files:
        img = cv2.imread(os.path.join(in_, name), 0)
        cv2.imwrite(os.path.join(out_, name), img)
    print('done convert to gray')


def resize(in_, out_, scale=0.5):
    print('start resizing')
    files = os.listdir(in_)
    for name in files:
        img = cv2.imread(os.path.join(in_, name))
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        cv2.imwrite(os.path.join(out_, name), img)
    print('done resizing')


def check_anomalies(angles_average):
    # function to calculate how many points changed their location
    # kol_min_90 - parameter that calculates how many times angle was less than threshold
    # num_decrease_average - parameter that calculates how many times angle was closer towards 90 degrees
    # number_of_zeros_sec - parameter that calculates how many times angle was close to 0 degree
    num_decrease_average = 0
    kol_min_90 = 0
    threshold = 0.3  # how much the angle can be smaller
    threshold_for_others = 1
    percents_second_person_appears = 0  # .8
    number_of_others = 0
    current_angle_average = min(abs(angles_average[0][0] - rad_270), abs(angles_average[0][0] - rad_90))
    number_of_zeros_sec = 0
    for i in range(0, time_period_sleep):
        for j in range(1, MAX_PERSONS):
            other_angle = min(abs(angles_average[0][0] - rad_270), abs(angles_average[0][0] - rad_90))
            if other_angle <= threshold_for_others:
                number_of_others += 1
        detected_angle = min(abs(angles_average[0][i]), abs(angles_average[0][i] - rad_180))
        if detected_angle <= threshold:
            number_of_zeros_sec += 1
    for i in range(time_period_sleep, time_period):
        for j in range(1, MAX_PERSONS):
            other_angle = min(abs(angles_average[0][0] - rad_270), abs(angles_average[0][0] - rad_90))
            if other_angle <= threshold_for_others:
                number_of_others += 1
        new_angle_average = min(abs(angles_average[0][i] - rad_270), abs(angles_average[0][i] - rad_90))
        if new_angle_average < current_angle_average:
            num_decrease_average += 1
            current_angle_average = new_angle_average
        # if there are some angles which are very close to 90 degrees (or 270 degree)
        if new_angle_average <= threshold:
            kol_min_90 += 1

    if number_of_zeros_sec >= 0.6 * time_period_sleep and num_decrease_average >= 3 and kol_min_90 >= 5 and number_of_others >= percents_second_person_appears * time_period:
        return True
    else:
        return False


def convert_to_skelets(in_, out_, cpu=False, height_size=256):
    #   height_size - network input layer height size
    #   cpu - True if we would like to run in CPU
    print('start convert to skelets')
    # mask that shows - this is bed
    mask = cv2.imread(os.path.join('mask', 'mask.jpg'), 0)
    mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    net = PoseEstimationWithMobileNet()

    load_state(net, checkpoint)
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4

    max_number = 963
    num_img = 0
    stream = cv2.VideoCapture("rtsp://admin:cj,frfeks,frf@62.140.233.76:554")
#    for num in range(0, max_number + 1):
    while(True):
#        frame = 'frame' + str(num) + '.jpg'
#        img = cv2.imread(os.path.join(in_, frame), cv2.IMREAD_COLOR)

        r, img = stream.read()

        # cv2.destroyAllWindows()
        # find the place of the bed - and add border to it, so we can cut the unnecessary part
        # apply object detection and find bed
        # output is an image with black pixels of not bed, and white pixels of bed

        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        # how many persons in image
        num_persons = len(pose_entries)
        # num_img more than time_period - we delete first second and add the last second


        bones_detected = np.zeros(len(bones_to_detect))
        bones_xa = np.zeros(len(bones_to_detect))
        bones_ya = np.zeros(len(bones_to_detect))
        bones_xb = np.zeros(len(bones_to_detect))
        bones_yb = np.zeros(len(bones_to_detect))
        bones_in_bed = np.zeros(len(bones_to_detect))

        for n in range(num_persons):
            count_person_not_in_bed = 1
            for id_x in range(len(bones_to_detect)):
                bones_detected[id_x] = 0
                bones_xa[id_x] = 0
                bones_ya[id_x] = 0
                bones_xb[id_x] = 0
                bones_yb[id_x] = 0
                bones_in_bed[id_x] = 0
            if len(pose_entries[n]) == 0:
                continue
            for id_, part_id in enumerate(bones_to_detect):
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                global_kpt_a_id = pose_entries[n][kpt_a_id]
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                global_kpt_b_id = pose_entries[n][kpt_b_id]
                # if both points are detected
                if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                    bones_xa[id_], bones_ya[id_] = all_keypoints[int(global_kpt_a_id), 0:2]
                    bones_xb[id_], bones_yb[id_] = all_keypoints[int(global_kpt_b_id), 0:2]
                    if mask[int(bones_ya[id_])][int(bones_xa[id_])] == 1 and mask[int(bones_yb[id_])][int(bones_xb[id_])] == 1:
                        bones_in_bed[id_] = 1
                    bones_detected[id_] = 1

            sum_bones = 0
            for id_, val in enumerate(bones_in_bed):
                sum_bones += val
            if sum_bones == len(bones_in_bed):
                # anomaly
                # we take mean vector of 2 vectors of bones 6 and 9
                bone_xa = (bones_xa[0] + bones_xa[2])/2
                bone_ya = (bones_ya[0] + bones_ya[2])/2
                bone_xb = (bones_xb[0] + bones_xb[2])/2
                bone_yb = (bones_yb[0] + bones_yb[2])/2
                x1 = bone_xb - bone_xa
                y1 = bone_yb - bone_ya
                x2 = 100
                y2 = 0
                global anomaly_checker
                alfa = math.acos(
                    (x1 * x2 + y1 * y2) / (math.sqrt(x1 ** 2 + y1 ** 2) * math.sqrt(x2 ** 2 + y2 ** 2)))
                # if alfa is close to 90 degree - anomaly
                if min(abs(alfa - rad_90), abs(alfa - rad_270)) <= threshold:
                    print('num_persons', num_persons)
                    if num_persons == 1:
                        anomaly_checker = np.delete(anomaly_checker, 0)
                        anomaly_checker = np.append(anomaly_checker, 1)
                    cv2.imwrite(os.path.join('out_out', frame), img)
                if np.sum(anomaly_checker) >= SEC_WITHOUT_HELP:
                    print('ALARM!')

        num_img += 1

        if not os.path.exists(out_):
            os.mkdir(out_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('done convert to skelets')


def create_video_from_photos(in_folder, out_video):
    frame = cv2.imread(os.path.join(in_folder, 'frame1.jpg'))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))
    num_files = 589
    plt.figure(figsize=[15, 15])

    for num in range(1, num_files + 1):
        frame = 'frame' + str(num) + '.jpg'
        img = cv2.imread(os.path.join(in_folder, frame))

        try:
            img = cv2.resize(img, (width, height))
            video.write(img)
        except Exception as e:
            print(e)
    cv2.destroyAllWindows()
    video.release()
    print('done creating video')


def create_image_mask(in_):
    print('start creation of mask')
    # create mask from existing image of bed
    # if bed is detected - we return result to folder mask
    # the result image is in folder /mask/mask.jpg
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    # load image where bed should be detected
    image = skimage.io.imread(os.path.join(in_, 'frame0.png'))

    # Run detection
    results = model.detect([image], verbose=1)

    r = results[0]
    masks = []
    scores = []
    size_of_mask = []
    # Number of instances
    N = r['rois'].shape[0]
    for i in range(0, N):
        # Mask
        if r['class_ids'][i] == 60 or r['class_ids'][i] == 58: #class is bed
            masks.append(r['masks'][:, :, i])
            scores.append(r['scores'][i])
            size_of_mask.append(np.sum(np.where(masks[-1] == 1, 1, 0)))
    max_id = np.argmax(size_of_mask)
    mask = masks[max_id]
    blank_image = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    #image = cv2.imread('desktop.png')#os.path.join('out_16', '67.jpg'), 0)
    blank_image[:, :, 0] = np.where(mask == 1,
                              255,
                              #image[:, :, 0])
                              blank_image[:, :, 0])

    cv2.imwrite('mask/mask.jpg', blank_image)
    cv2.waitKey(0)
    print('mask created')


def connect_to_webcam():

    stream = cv2.VideoCapture("rtsp://admin:cj,frfeks,frf@62.140.233.76:554")

    while True:

        r, frame = stream.read()
        cv2.imshow('IP Camera stream', frame)
        cv2.imwrite('mask/frame0.png', frame)
        create_image_mask('mask')
        print('done')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    #connect_to_webcam()
    #convert_video_to_pics('17.mp4', '17')
    #    convert_to_gray('out', 'out_2')
    #    resize('out_2', 'out_3', 0.3)
    #create_image_mask('17')
    convert_to_skelets('17', 'out_17')
    # create_video_from_photos('out_5', 'output.avi')


if __name__ == "__main__":
    main()
