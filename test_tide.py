# The bare-bones example from the README.
# Run coco_example.py first to get mask_rcnn_bbox.json
from tidecv import TIDE, datasets

tide = TIDE()
# city2foggy_source_only
# tide.evaluate_range(datasets.COCO("/data1/TL/data/Cityscapes/annotations/instances_val2017.json"), datasets.COCOResult('/data1/lsg/res_cocotype_json/city2foggy_source_only__0.0001_1e-05_0.002.json'), mode=TIDE.BOX)

# city2bdd_source_only
# tide.evaluate_range(datasets.COCO("/data1/TL/data/bdd100k/bdd100k_labels_images_det_coco_val.json"), datasets.COCOResult('/data1/lsg/res_cocotype_json/city2bdd_source_only__0.00028_2.8e-05_0.002.json'), mode=TIDE.BOX, task="city2bdd")

# sim2city_source_only
tide.evaluate_range(datasets.COCO("/data1/TL/data/Cityscapes/annotations/instances_val2017_caronly.json"), datasets.COCOResult('/data1/lsg/res_cocotype_json/sim2city_source_only__0.0001_1e-05_0.002.json'), mode=TIDE.BOX)

tide.summarize()
tide.plot()
