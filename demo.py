from __future__ import absolute_import

from got10k.experiments import *
import os
import cv2
from siamfc_VGG_cf_add import TrackerSiamFC
import time
import glob
import argparse

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

if __name__ == '__main__':
    # setup tracker
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--video_name', default='', type=str, help='videos or image files', required=True)
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()
    net_path = args.model
    tracker = TrackerSiamFC(net_path=net_path)
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    time_whole = 0
    frame_number = 0
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            frame_number += 1
            begin = time.time()
            bb = tracker.update(frame)
            end = time.time()
            time_whole += (end - begin)
            bb = list(map(int, bb))
            cv2.rectangle(frame,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),(0,255,255),1)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)    
    print("End!! The FPS of the tracker is {}".format(frame_number / time_whole))

