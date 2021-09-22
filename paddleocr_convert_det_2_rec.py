'''
把标注好的文本检测数据转化到文本识别数据
'''
import json
import os
import argparse

import cv2
import numpy as np
from scipy.spatial import distance as dist
from tqdm import tqdm

def order_points_old(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def order_points(pts):
    '''
    From https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    '''
	# sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
	# x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_perspective_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    #     print(pts)
    rect = order_points_old(pts)
    (tl, tr, br, bl) = rect
    #     print(rect)
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )
    #     print(dst)
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def make_rec_data(
    base_path ,
    txt_name,
    to_base_path,
    to_rec_name,
        ):
    '''
    将拿到的标注后的集装箱号数据，转换成ocr text recognition模型能够使用的格式
    '''
    txt_path = os.path.join(base_path, txt_name)

    to_txt_path = os.path.join(to_base_path,to_rec_name + ".txt")
    to_img_path = os.path.join(to_base_path,to_rec_name)
    os.makedirs(to_img_path,exist_ok=True)

    global_id = 0
    new_lines = []
    with open(txt_path,'r') as f:
        for line in tqdm(list(f)):
            per_line = line.split("\t")

            sub_path = per_line[0]
            labels = json.loads(per_line[1].strip())
            
            img = cv2.imread(os.path.join(base_path,sub_path))

            if img is None:
                continue

            for label in labels:
                _img = img.copy()
                transcription = label['transcription']
                points = np.array(label['points'],dtype = np.int32)

                text_area = four_point_perspective_transform(_img,np.array(points))

                _name = f"{global_id}.jpg"
                cv2.imwrite(os.path.join(to_img_path,_name),text_area)

                temp_path = os.path.join(to_rec_name.split(".")[0],_name)
                _line = temp_path + '\t' + transcription + '\n'
                new_lines.append(_line)
                # print(_line)

                global_id += 1

                # print(global_id)
    
    with open(to_txt_path,'w') as ft:
        ft.writelines(new_lines)

def get_args():
    args = argparse.ArgumentParser("把检测标注的数据转成识别标注的数据")
    args.add_argument(
        "--det_base_path",
        type=str, 
        default="2021_8_9/labeled",
    )
    args.add_argument(
        "--det_txt_name",
        type=str,
        default="rizhao_zhakou_p1.txt",
    )
    args.add_argument(
        "--to_rec_base_path", 
        type=str, 
        default="2021_8_9/labeled_ocr_rec/",
    )
    args.add_argument(
        "--to_rec_name", 
        type=str, 
        default="rizhao_zhakou_ocr_rec_p1",
    )

    return args

if __name__ == "__main__":
    args = get_args().parse_args()
    make_rec_data(args.det_base_path,args.det_txt_name,args.to_rec_base_path,args.to_rec_name)