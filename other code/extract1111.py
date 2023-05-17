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


def extractwatermark1(watermarkedimage, originalimage, X_watermark, Y_watermark):       
 
 finalwatermark = np.zeros((X_watermark,Y_watermark,3), 'uint')  
 print(type(originalimage))
 print(originalimage.dtype)
 print(type(watermarkedimage))
 print(watermarkedimage.dtype)
 
 imageio.imwrite('100.jpg',watermarkedimage)
 
 imageio.imwrite('200.jpg',originalimage)
 
 r1_dct=cv2.dct(watermarkedimage[:,:,0])
 g1_dct=cv2.dct(watermarkedimage[:,:,1])
 b1_dct=cv2.dct(watermarkedimage[:,:,2])

    #dct for original image  
    
 dct_r=cv2.dct(originalimage[:,:,0])
 dct_g=cv2.dct(originalimage[:,:,1])
 dct_b=cv2.dct(originalimage[:,:,2])

 x=X_watermark;
 y=Y_watermark;
 
 a= (r1_dct[1:x+1,1:y+1] - dct_r[1:x+1,1:y+1] * alpha)  / (1-alpha) 
 b= (g1_dct[1:x+1,1:y+1] - dct_g[1:x+1,1:y+1] * alpha)  / (1-alpha)
 c= (b1_dct[1:x+1,1:y+1] - dct_b[1:x+1,1:y+1] * alpha)  / (1-alpha)
  
 finalwatermark[:,:,0]=cv2.idct(a)*255
 finalwatermark[:,:,1]=cv2.idct(b)*255
 finalwatermark[:,:,2]=cv2.idct(c)*255
#  finalwatermark = np.zeros((512,512,3),dtype=np.float64)

#  finalwatermark[:,:,0] = cv2.idct(a).astype(np.float64)*255
#  finalwatermark[:,:,1] = cv2.idct(b).astype(np.float64)*255
#  finalwatermark[:,:,2] = cv2.idct(c).astype(np.float64)*255

 imageio.imwrite('lastimg.jpg',finalwatermark)
 print(type(finalwatermark))
 print(finalwatermark.dtype)
 
#  finalwatermark = np.uint8(finalwatermark)

#  print(type(finalwatermark))
#  print(finalwatermark.dtype)
 
#  final_watermark_normalized = cv2.normalize(finalwatermark, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


#  rgb_image6 = cv2.cvtColor(final_watermark_normalized, cv2.COLOR_BGR2RGB)
 
#  print(type(rgb_image6))
#  print(rgb_image6.dtype)
 
#  cv2.imwrite('hello.jpg',rgb_image6)
 return (finalwatermark)

alpha=0.99

cover = 'C:/Users/dines/Desktop/afterexam/girl.jpg'
coverimage=performzeropadding( imageio.imread(cover)) #read image in RGB
coverimage=coverimage/255;
water= 'C:/Users/dines/Desktop/project 2/static/image/watermarked.jpg'
watermarkedimage =performzeropadding( imageio.imread(water)) #read image in RGB
watermarkedimage=watermarkedimage/255;

print(type(coverimage))
print(coverimage.dtype)
print(type(watermarkedimage))
print(watermarkedimage.dtype)

finalwatermark = extractwatermark1(watermarkedimage, coverimage, 512, 512)
print(type(finalwatermark))
print(finalwatermark.dtype)
# plt.imshow(waterMarkImgExtracted)
# plt.title('Extracted watermark Image')
# plt.show()
# cv2.imwrite('waterMarkImgExtracted.jpg',waterMarkImgExtracted*255)
finalwatermark = np.uint8(finalwatermark)
final_watermark_normalized = cv2.normalize(finalwatermark, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


rgb_image6 = cv2.cvtColor(final_watermark_normalized, cv2.COLOR_BGR2RGB)
cv2.imwrite('waterMarkImgExtracted.jpg',rgb_image6)