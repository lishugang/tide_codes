from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_json_file', default='/data1/TL/data/Cityscapes/annotations/instances_val2017.json', type=str)
parser.add_argument('--pre_json_file', default='/data1/lsg/res_cocotype_json/city2foggy_source_only__0.0001_1e-05_0.002.json', type=str)
args = parser.parse_args()


clss ={'1':'person',  '2':'car' ,  '3':'bicycle' ,  '4':'rider' ,  '5':'truck' ,  '6':'bus' ,  '7':'motorcycle' ,  '8':'train'}
# clss ={'1':'person',  '2':'car' ,  '3':'bicycle' ,  '4':'rider' ,  '5':'truck' ,  '6':'bus' ,  '7':'motorcycle' }
# clss ={'1':'car'}
coco_obj = COCO(args.annotation_json_file)

dets = coco_obj.loadRes(args.pre_json_file)

coco_eval = COCOeval(coco_obj, dets, "bbox")

                                          #
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

res = []
for i in range(len(clss.keys())):
    coco_eval.params.catIds = [i+1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("class : ", clss[str(i+1)])
    
