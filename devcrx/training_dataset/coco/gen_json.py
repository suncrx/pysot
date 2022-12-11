#Genareate json file
import os.path

from pycocotools.coco import COCO
from os.path import join
import json

#--------------------------------------------
#dataDir = '.'
dataDir = 'D:/GeoData/Benchmark/COCO'
dataTypes = ['val2014', 'train2014']
outDir = os.path.join(dataDir, 'crop511')
#------------------------------------

for dataType in dataTypes:
    dataset = dict()
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    n_imgs = len(coco.imgs)
    for n, img_id in enumerate(coco.imgs):
        print('subset: {} image id: {:04d} / {:04d}'.format(dataType, n, n_imgs))
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        video_crop_base_path = join(dataType, img['file_name'].split('/')[-1].split('.')[0])
        
        if len(anns) > 0:
            dataset[video_crop_base_path] = dict()        

        for trackid, ann in enumerate(anns):
            rect = ann['bbox']
            c = ann['category_id']
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue
            dataset[video_crop_base_path]['{:02d}'.format(trackid)] = {'000000': bbox}

    print('Saving json (dataset), please wait ...')
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    outFile = os.path.join(outDir, '{}.json'.format(dataType))
    json.dump(dataset, open(outFile, 'w'), indent=4, sort_keys=True)
    print('Json file saved: ' + outFile)
    print('done!')

