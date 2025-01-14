import os.path
from os.path import join
from os import listdir
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET


VID_base_path = 'D:/GeoData/Benchmark/imagenet-mini'
ann_base_path = join(VID_base_path, 'Annotations/train/')
img_base_path = join(VID_base_path, 'Data/train/')
#sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})

visual = True
delay = 1000
color_bar = np.random.randint(0, 255, (90, 3))

stop = False
sub_set_base_path = join(ann_base_path, '')
videos = sorted(listdir(sub_set_base_path))
for vi, video in enumerate(videos):
    print('video id: {:04d} / {:04d}'.format(vi, len(videos)))

    video_base_path = join(sub_set_base_path, video)
    xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
    for xml in xmls:
        # search the corresponding jpeg file
        jpg_fp = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')
        # if the jpeg file does not exist, skip
        if not os.path.exists(jpg_fp):
            continue

        f = dict()
        xmltree = ET.parse(xml)
        size = xmltree.findall('size')[0]
        frame_sz = [int(it.text) for it in size]
        objects = xmltree.findall('object')
        if visual:
            im = cv2.imread(jpg_fp)
        for object_iter in objects:
            #trackid = int(object_iter.find('trackid').text)
            trackid = 0
            # get bbox from the xml file
            bndbox = object_iter.find('bndbox')
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            if visual:
                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[2]), int(bbox[3]))
                #cv2.rectangle(im, pt1, pt2, color_bar[trackid], 2)
                cv2.rectangle(im, pt1, pt2, [0,255,0], 2)
        if visual:
            cv2.imshow('img', im)
            k = cv2.waitKey(delay)
            if k == 27:
                stop = True
                break
    if stop:
        break

print('done!')
