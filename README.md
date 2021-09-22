# Tools

## Keywords

Deeplearning, Nvidia, Graphics card, cache, scripts

## About

Accumulate some common scripts

# Introduction and usage

## clear_vanish_cache.py

### About

Sometimes, When our process exits, there will still be residual graphics memory on some used graphics cards. And sometimes we can not get pid from command "nvidia-smi". Now we need command fuser to kill the processes and clear the residual cache.

### Usage

```shell
${/absolute/path/of/python/interpreter} clear_vanish_cache.py ${graphics card number} 
```

or be careful

```shell
sudo ${/absolute/path/of/python/interpreter} clear_vanish_cache.py ${graphics card number} 
```

## image_augmentation/add_shadow.py

### About

Simulate the shadow effect and add it to photo.
Learn from https://www.freecodecamp.org/news/image-augmentation
### usage:

```python
img = cv2.imread("/path/to/image")
shadow_aug = Shadow_aug()
img_aug = shadow_aug(img)
cv2.imwrite("/path/to/image_shadowed",img_aug)
```

## draw_coco_segmentation_anno.py

### About

Given COCO format segmentation annotations and photos, the tool draw bbox, segmentation polygon and segmentation mask on the given image.
### usage:

```shell
python draw_coco_segmentation_anno.py --img_dir=/path/to/images --anno_file=/path/to/annotations.json --to_dir=/path/to/save/drawed
```

## paddleocr_convert_det_2_rec.py

### 

For text detection data that follow paddleOCR data format, the tool convert text detection data to text recognition data that follow paddeOCR data format.

### usage

```python 
paddleocr_convert_det_2_rec.py --det_base_path=/path/to/det/data --det_txt_name=/path/to/det/label --to_rec_base_path=/path/to/save/rec/data --to_rec_name=/path/to/save/rec/label
```