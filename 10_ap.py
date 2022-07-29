# The bare-bones example from the README.
# Run coco_example.py first to get mask_rcnn_bbox.json
from tidecv import TIDE, datasets

tide = TIDE()
# city2foggy_source_only
# tide.evaluate_range(datasets.COCO("/data1/TL/data/Cityscapes/annotations/instances_val2017.json"), datasets.COCOResult('source_city2foggy.json'), thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],mode=TIDE.BOX)
# tide.summarize()

# city2foggy_sfa
# tide.evaluate_range(datasets.COCO("/data1/TL/data/Cityscapes/annotations/instances_val2017.json"), datasets.COCOResult('sfa_city2foggy.json'),thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], mode=TIDE.BOX)
# tide.summarize()

# city2foggy_ours
tide.evaluate_range(datasets.COCO("/data1/TL/data/Cityscapes/annotations/instances_val2017.json"), datasets.COCOResult('ours_city2foggy.json'), thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],mode=TIDE.BOX)

# city2bdd_source_only
# tide.evaluate_range(datasets.COCO("/data1/TL/data/bdd100k/bdd100k_labels_images_det_coco_val.json"), datasets.COCOResult('/data1/lsg/res_cocotype_json/city2bdd_source_only__0.00028_2.8e-05_0.002.json'), mode=TIDE.BOX, task="city2bdd")

# sim2city_source_only
# tide.evaluate_range(datasets.COCO("/data1/TL/data/Cityscapes/annotations/instances_val2017_caronly.json"), datasets.COCOResult('/data1/lsg/res_cocotype_json/sim2city_source_only__0.0001_1e-05_0.002.json'), mode=TIDE.BOX)


# tide.evaluate_range(datasets.COCO("/data1/TL/data/Cityscapes/annotations/instances_val2017.json"), datasets.COCOResult('DA_decoder_4e3_source2DA.json'),thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], mode=TIDE.BOX)

tide.summarize()
# tide.plot()
