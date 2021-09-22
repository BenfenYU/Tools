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
Learn from https://www.freecodecamp.org/news/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f/

### usage:

```python
img = cv2.imread("/path/to/image")
shadow_aug = Shadow_aug()
img_aug = shadow_aug(img)
cv2.imwrite("/path/to/image_shadowed",img_aug)
```