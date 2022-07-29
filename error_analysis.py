import json
import time
from operator import itemgetter
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

class Error_Analysis:
    def __init__(self, gt_path, dt_path,num, method_task):
        self.gt_path = gt_path
        self.dt_path = dt_path
        self.iouType = 'bbox'
        # self.total = 20000
        self.total = num
        self.correct = 0
        self.mis_localized = 0
        self.background = 0
        self.plot_title = dt_path.split('/')[-1].split('.json')[0]
        self.count = -1
        self.method_task = method_task

    def data_prepare(self):
        with open(self.dt_path,'r') as f:
            res = json.load(f)
        res_sorted = sorted(res, key = itemgetter('score'), reverse = True)
        if len(res_sorted) > self.total:
            res_2w = res_sorted[:self.total]
            with open('dt_2w.json', 'w') as j:
                json.dump(res_2w, j)
                self.dt_path = 'dt_2w.json'
        else:
            self.total = len(res)
            print('val_obj <= 20000')
            time.sleep(5)
        self.cocoGt = COCO(self.gt_path)
        
        self.cocoDt = self.cocoGt.loadRes(self.dt_path)

        self.gt_catIds = self.cocoGt.getCatIds()
        self.gt_imgIds = self.cocoGt.getImgIds()
        # print(len(self.gt_catIds))  # 8
        # print(len(self.gt_imgIds))  # 500
        self.dt_catIds = []
        self.dt_imgIds = []
        self.catToImgs = defaultdict(set)
        with open(self.dt_path,'r') as d:
            dt_json = json.load(d)
        # print(len(dt_json))   #20000
        for dt_obj in dt_json:
            # print(dt_obj)
            self.dt_catIds.append(dt_obj['category_id'])
            self.dt_imgIds.append(dt_obj['image_id'])
            self.catToImgs[dt_obj['category_id']].add(dt_obj['image_id'])
        self.dt_catIds = list(set(self.dt_catIds))
        self.dt_imgIds = list(set(self.dt_imgIds))
        # print(len(self.dt_catIds))  # 8
        # print(len(self.dt_imgIds))  # 493
        # print(len(self.catToImgs.keys()))   # 8
        # i = 0
        # for key in self.catToImgs.keys():
        #     i+=len(self.catToImgs[key])
        # print(i)   # 2223
        # time.sleep(50)

        self.gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.gt_imgIds))
        self.dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=self.dt_imgIds))
        # print(self.dts[0])
        # print(self.gts[0])
        # print(len(self.gts))   # 9792
        # print(len(self.dts))   # 20000
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)  
        # 给对应img，类别 添加对应的bbox信息
        for gt in self.gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in self.dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        # print(len(self._gts))   # 1718
        # print(len(self._dts))   # 2223
        # time.sleep(50)
        # i = 0
        # for key in self._dts.keys():
        #     i+=len(self._dts[key])
        # print(i)   # 20000
        # time.sleep(50)

        
    
    def computeIoU(self, gt, dt): 

        
        # 按照网络预测的置信度score排序
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        # print(len(dt))
        # time.sleep(50)
        if self.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
            # print(g)
            # print("***************")
            # print(d)  
            # time.sleep(1)     
        else:
            raise Exception('unknown iouType for iou computation')
        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        # print(iscrowd)
        ious = maskUtils.iou(d,g,iscrowd)
        return np.array(ious)


    def run(self):
        i = 0
        cls_inner = 0
        self.data_prepare()
        for catId in self.dt_catIds:
            # print(catId)
            imgIds = self.catToImgs[catId]
            # print(imgIds)
            for imgId in imgIds:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
                # print(len(dt))
                # print(len(gt))   # 1
                # print(len(dt))   # 6
                if len(gt) == 0 or len(dt) == 0:
                    continue
                ious = self.computeIoU(gt, dt)
                cls_inner += ious.shape[0]
                # print(np.array(ious).shape)
                i += 1
                # time.sleep(1)
                # print(ious)
                ious_ = np.max(ious, axis=1)
                # print(ious_)
                self.correct += np.sum(ious_>=0.5) 
                self.background += np.sum(ious_<=0.3)
        self.background += self.total - cls_inner
        self.mis_localized = self.total - self.background - self.correct
        print('correct = {0}, mis_localized = {1}, background = {2}'.format(self.correct, self.mis_localized, self.background))
        # print(i)  # 2223
        # print(cls_inner)  # 18205 不足2W是因为有的检测出来了，但是gt中不存在那一类

    
    def func(self, pct, _labels):
        self.count += 1
        return "{}:\n{:3.1f}%".format(_labels[self.count], pct)


    def plot(self):
        plt.figure(figsize=(6,6))
        labels = ['correct', 'mis_localized', 'background']
        _labels = ['Cor', 'Misloc', 'BG']
        colors = ['palegreen', 'lightskyblue', 'peachpuff']
        values = [self.correct, self.mis_localized, self.background]
        plt.pie(values, colors=colors, autopct = lambda pct: self.func(pct, _labels), textprops= {'fontsize':15,'color':'black'})
        plt.title(self.plot_title, y=-0.01, fontsize = 15)
        plt.legend(labels = labels, loc = "best", frameon=False,  )
        print('saving ...')
        plt.savefig(self.method_task +str(self.total)+'.png')
        plt.show()



if __name__ == "__main__":
    gt_path = '/data1/TL/data/Cityscapes/annotations/instances_val2017.json'
    dt_path = 'source_city2foggy.json'
    num = 8000
    method_task = 'source_city2foggy_'
    error_analysis = Error_Analysis(gt_path, dt_path, num, method_task )
    
    error_analysis.run()
    error_analysis.plot()




