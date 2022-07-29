from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import argparse


coco_obj = COCO('/data1/TL/data/Cityscapes/annotations/instances_val2017.json')
ids = coco_obj.getCatIds()
print(ids)
dets = coco_obj.loadRes('sfa_city2foggy.json')

coco_eval = COCOeval(coco_obj, dets, "bbox")


coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()










