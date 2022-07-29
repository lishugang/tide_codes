# The bare-bones example from the README.
# Run coco_example.py first to get mask_rcnn_bbox.json
from tidecv import TIDE, datasets


gt = '/data1/TL/data/Cityscapes/annotations/instances_val2017.json'
dt = 'ours_city2foggy.json'

tide = TIDE()
tide.evaluate_range(datasets.COCO(gt), datasets.COCOResult(dt), mode=TIDE.BOX)

tide.summarize()
tide.plot()
