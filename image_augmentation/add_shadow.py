"""
Learn from https://www.freecodecamp.org/news/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f/

usage:

img = cv2.imread("/path/to/image")
shadow_aug = Shadow_aug()
img_aug = shadow_aug(img)
cv2.imwrite("/path/to/image_shadowed",img_aug)
"""

import random
import numpy as np
import cv2

class Shadow_aug():
        
    def generate_shadow_coordinates(self,imshape,number_shadows,points_number_per_shadow):    
        vertices_list=[]    
        for _ in range(number_shadows):        
            vertex=[]        
            for _ in range(np.random.randint(*points_number_per_shadow)): 
                ## Dimensionality of the shadow polygon            
                vertex.append(( imshape[1]*np.random.uniform(),imshape[0]*np.random.uniform()))        
                vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices         
            vertices_list.append(vertices)    
        return vertices_list ## List of shadow vertices

    def __call__(self,image,
                    number_shadows = 3,
                    points_number_per_shadow = (5,15),
                    shadow_light_range = (0.4,0.6),):    
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
        mask = np.zeros_like(image)     
        imshape = image.shape    
        vertices_list= self.generate_shadow_coordinates(imshape,number_shadows,points_number_per_shadow) #3 getting list of shadow vertices    
        for vertices in vertices_list:         
            cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel        
            shadow_light = random.uniform(*shadow_light_range)
            image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*shadow_light   ## if red channel is hot, image's "Lightness" channel's brightness is lowered     
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
        return image_RGB