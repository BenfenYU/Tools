import os
from pycocotools.coco import COCO
import numpy as np
import cv2
import shutil
import copy
import argparse
from utils import if_exist_rm_then_make
from tqdm import tqdm

def if_exist_rm_then_make(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def if_exist_rm_then_make(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def xywh_to_xyxy(bboxes):
    bboxes = copy.deepcopy(bboxes)
    for bbox in bboxes:
        bbox[2] += bbox[0] 
        bbox[3] += bbox[1] 
    
    return bboxes

def xyxy_to_xywh(bboxes):
    bboxes = copy.deepcopy(bboxes)
    for bbox in bboxes:
        bbox[2] -= bbox[0] 
        bbox[3] -= bbox[1] 
    
    return bboxes

def min_bbox(bboxes):
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes,dtype = np.int32)
    
    xyxy_boxes = xywh_to_xyxy(bboxes)
    x1 = int(np.min(xyxy_boxes[:,0]))
    y1 = int(np.min(xyxy_boxes[:,1]))
    x2 = int(np.max(xyxy_boxes[:,2]))
    y2 = int(np.max(xyxy_boxes[:,3]))
    
    return xyxy_to_xywh([[x1,y1,x2,y2]])[0]

def arrange_segmentation_points(segm_raw_points):
    '''
    把coco格式的分割点，转换成xy坐标形式的点
    '''
    arranged_points = []
    for sub_raw_points in segm_raw_points:
        temp_points = []
        for x,y in zip(sub_raw_points[::2],sub_raw_points[1::2]):
            temp_points.append([x,y])
        
        arranged_points.append(temp_points)
    
    return arranged_points

def draw_segmentation_polygon(img,segmentations,color = (255, 0,0),thickness = 5):
    img = img.copy()
    for segm in segmentations:
        segm_raw_points = segm["segmentation"]
        arranged_points = arrange_segmentation_points(segm_raw_points)
        for sub_arranged_points in arranged_points:
            points = np.array(sub_arranged_points,dtype = np.int32)
            img = cv2.polylines(img,[points] ,1, color,thickness)
    
    return img

def draw_bbox(img,segmentations,color = (255, 0,0),thickness = 5):
    img = img.copy()
    for segm in segmentations:
        bbox = segm["bbox"]
        bbox = np.array(bbox,dtype = np.int32)
        x1,y1,x2,y2 = bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1] + bbox[3]
        img = cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness)
    
    return img

def draw_mask(img,segmentations,color = (255, 0,0),thickness = 5):
    from imgaug.augmentables.polys import Polygon

    img = img.copy()
    for segm in segmentations:
        segm_raw_points = segm["segmentation"]
        arranged_points = arrange_segmentation_points(segm_raw_points)
        for sub_arranged_points in arranged_points:
            points = np.array(sub_arranged_points,dtype = np.int32)
            polygon = Polygon(points)
            img = polygon.draw_on_image(img,color,alpha_face=0.4)
    
    return img

def draw_w_image_json_dir():
    img_dir,anno_file,to_dir = args.img_dir,args.anno_file,args.to_dir
    if_exist_rm_then_make(to_dir)

    coco = COCO(anno_file)
    img_ids = list(coco.imgs.keys())
    for img_id in tqdm(img_ids):
        file_name = coco.imgs[img_id]["file_name"]
        img_path = os.path.join(img_dir,file_name)
        img = cv2.imread(img_path)
        ann_ids = coco.getAnnIds(img_id)
        segmentations = coco.loadAnns(ann_ids)

        drawed_img = draw_segmentation_polygon(img,segmentations)
        drawed_img = draw_mask(drawed_img,segmentations)
        drawed_img = draw_bbox(drawed_img,segmentations)
        cv2.imwrite(os.path.join(to_dir,f"{file_name}"),drawed_img)

if __name__ == "__main__":
    '''
    给出图片目录和COCO格式的json位置，把标注画上去
    '''

    parser = argparse.ArgumentParser(
        description='Split fresh data script')

    parser.add_argument('--img_dir', default="/mnt/ext_data/jingbo/data/2021_7_23_merge_anns/instance_container_merged_data/images", type=str,
                        help='')
    parser.add_argument('--anno_file', default="/mnt/ext_data/jingbo/data/2021_7_23_merge_anns/instance_container_merged_data/train.json", type=str,
                        help='')
    parser.add_argument('--to_dir', default="temp", type=str,
                        help='')

    parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
    
    global args
    args = parser.parse_args()

    draw_w_image_json_dir()