#Visualize COCO train and val image and annotations.

import os.path
from pycocotools.coco import COCO
import cv2
import numpy as np

#------------------------------------------------------------------------
dataDir = 'D:/GeoData/Benchmark/COCO'
dataType = 'val2014'
#dataType = 'train2014'

visual = True

#------------------------------------------------------------------------
annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)

color_bar = np.random.randint(0, 255, (90, 3))

for img_id in coco.imgs:
    img = coco.loadImgs(img_id)[0]
    img_path = '{}/{}/{}'.format(dataDir, dataType, img['file_name'])
    if not os.path.exists(img_path):
        continue

    im = cv2.imread(img_path)

    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        rect = ann['bbox']
        c = ann['category_id']
        if visual:
            pt1 = (int(rect[0]), int(rect[1]))
            pt2 = (int(rect[0]+rect[2]), int(rect[1]+rect[3]))
            cv2.rectangle(im, pt1, pt2, color_bar[c-1].tolist(), 2)
            #cv2.rectangle(im, pt1, pt2, [0,255,0], 2)
    cv2.imshow('img', im)
    k = cv2.waitKey(200)
    if k==27:
        break

print('done')

