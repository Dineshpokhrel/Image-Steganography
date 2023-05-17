import cv2
import numpy as np
import imageio.v2 as imageio

# cover=r"C:\Users\dines\Desktop\project 2\test_files\girl.jpg"
# water=r"C:\Users\dines\Desktop\project 2\static\image\watermarked.jpg"
# Load the watermarked image and the original cover image
originalimage = cv2.imread(r"C:\Users\dines\Desktop\project 2\test_files\girl.jpg")
watermarkedimage = cv2.imread(r"C:\Users\dines\Desktop\project 2\static\image\watermarked.jpg")

# Split the watermarked image and cover image into RGB components
b_wm, g_wm, r_wm = cv2.split(watermarkedimage)
b_cover, g_cover, r_cover = cv2.split(originalimage)

X_watermark=512
Y_watermark=512
alpha=0.99



finalwatermark = np.zeros((X_watermark,Y_watermark,3), 'uint')  

r1_dct=cv2.dct(watermarkedimage[:,:,2].astype(np.float64))
g1_dct=cv2.dct(watermarkedimage[:,:,1].astype(np.float64))
b1_dct=cv2.dct(watermarkedimage[:,:,0].astype(np.float64))
#dct for original image  
   
dct_r=cv2.dct(originalimage[:,:,2].astype(np.float64))
dct_g=cv2.dct(originalimage[:,:,1].astype(np.float64))
dct_b=cv2.dct(originalimage[:,:,0].astype(np.float64))
x=X_watermark;
y=Y_watermark;

a= (r1_dct[1:x+1,1:y+1] - dct_r[1:x+1,1:y+1] * alpha)  / (1-alpha) 
b= (g1_dct[1:x+1,1:y+1] - dct_g[1:x+1,1:y+1] * alpha)  / (1-alpha)
c= (b1_dct[1:x+1,1:y+1] - dct_b[1:x+1,1:y+1] * alpha)  / (1-alpha)
 
finalwatermark[:,:,2]=cv2.idct(c).astype('uint8')
finalwatermark[:,:,1]=cv2.idct(b).astype('uint8')
finalwatermark[:,:,0]=cv2.idct(a).astype('uint8')

watermark = cv2.merge((finalwatermark[:,:,2], finalwatermark[:,:,1], finalwatermark[:,:,0]))
watermark = np.mean(watermark, axis=2)

# finalwatermark_uint8 = finalwatermark.astype(np.uint8)
cv2.imwrite("aaaaaa.jpg", watermark)