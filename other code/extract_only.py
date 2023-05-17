import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import imageio.v2 as imageio
import math

def performzeropadding(image): 
 org_h,org_w,channel=np.array(image.shape)   
 org_h = np.int32(org_h)
 org_w = np.int32(org_w)
 

 nbh = math.ceil(org_h/8)            #number of block must be divisible by 8
 nbh = np.int32(nbh)

 nbw = math.ceil(org_w/8)
 nbw = np.int32(nbw)

 H =  8 * nbh
 W =  8 * nbw
    
 paddedimage = np.zeros((H,W,3), 'uint8')
 paddedimage[0:org_h,0:org_w] = image[0:org_h,0:org_w]
 return(paddedimage)

cover = 'C:/Users/dines/Desktop/afterexam/girl.jpg'
originalimage=performzeropadding( imageio.imread(cover)) #read image in RGB
water= 'C:/Users/dines/Desktop/project 2/static/image/watermarked.jpg'
watermarkedimage =performzeropadding( imageio.imread(water)) #read image in RGB
alpha=0.99
X_watermark=512
Y_watermark=512
finalwatermark = np.zeros((X_watermark,Y_watermark,3), 'uint')

cv2.imwrite('hey.jpg',watermarkedimage)

cv2.imwrite('hey1.jpg',originalimage)
#  watermarkedimage1 = cv2.convertScaleAbs(watermarkedimage)
#  rgb_image3 = cv2.cvtColor(watermarkedimage1, cv2.COLOR_BGR2RGB)
# imageio.imwrite('100.jpg',watermarkedimage)
 
#  originalimage1 = cv2.convertScaleAbs(originalimage)
#  rgb_image4 = cv2.cvtColor(originalimage1, cv2.COLOR_BGR2RGB)
# imageio.imwrite('200.jpg',originalimage)
 
# r1_dct=cv2.dct(watermarkedimage[:,:,0])
# g1_dct=cv2.dct(watermarkedimage[:,:,1])
# b1_dct=cv2.dct(watermarkedimage[:,:,2])
r1_dct=cv2.dct(watermarkedimage[:,:,0].astype(np.float64))
g1_dct=cv2.dct(watermarkedimage[:,:,1].astype(np.float64))
b1_dct=cv2.dct(watermarkedimage[:,:,2].astype(np.float64))

    #dct for original image  
    
# dct_r=cv2.dct(originalimage[:,:,0])
# dct_g=cv2.dct(originalimage[:,:,1])
# dct_b=cv2.dct(originalimage[:,:,2])
dct_r=cv2.dct(originalimage[:,:,0].astype(np.float64))
dct_g=cv2.dct(originalimage[:,:,1].astype(np.float64))
dct_b=cv2.dct(originalimage[:,:,2].astype(np.float64))

x=X_watermark;
y=Y_watermark;
 
a= (r1_dct[1:x+1,1:y+1] - dct_r[1:x+1,1:y+1] * alpha)  / (1-alpha) 
b= (g1_dct[1:x+1,1:y+1] - dct_g[1:x+1,1:y+1] * alpha)  / (1-alpha)
c= (b1_dct[1:x+1,1:y+1] - dct_b[1:x+1,1:y+1] * alpha)  / (1-alpha)
  
finalwatermark[:,:,0]=cv2.idct(a)*255
finalwatermark[:,:,1]=cv2.idct(b)*255
finalwatermark[:,:,2]=cv2.idct(c)*255
 
 #  normalizedfinalwatermark=np.clip(finalwatermark,0,1)
 
 #  cv2.imwrite('waterMarkImgExtracted.jpg',normalizedfinalwatermark*255)

 #  finalwatermark = np.uint8(finalwatermark)
 #  final_watermark_normalized = cv2.normalize(finalwatermark, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

 #  cv2.imwrite('waterMarkImgExtracted.jpg', cv2.UMat(final_watermark_normalized))
 
 
finalwatermark = np.uint8(finalwatermark)
final_watermark_normalized = cv2.normalize(finalwatermark, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


rgb_image6 = cv2.cvtColor(final_watermark_normalized, cv2.COLOR_BGR2RGB)
cv2.imwrite('hello.jpg',rgb_image6)