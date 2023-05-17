import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import imageio.v2 as imageio
import math

# def dct2(a):
#     return cv2.dct(cv2.dct(a.T, flags=cv2.DCT_ROWS).T, flags=cv2.DCT_ROWS)

# def idct2(a):
#     return cv2.idct(cv2.idct(a.T, flags=cv2.DCT_ROWS).T, flags=cv2.DCT_ROWS)

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


def extract_watermark1(watermarkedimage, originalimage, X_watermark, Y_watermark,alpha):       
 
 finalwatermark = np.zeros((X_watermark,Y_watermark,3), 'uint')  
 
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
 
 #  normalizedfinalwatermark=np.clip(finalwatermark,0,1)
 
 #  cv2.imwrite('waterMarkImgExtracted.jpg',normalizedfinalwatermark*255)

 #  finalwatermark = np.uint8(finalwatermark)
 #  final_watermark_normalized = cv2.normalize(finalwatermark, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

 #  cv2.imwrite('waterMarkImgExtracted.jpg', cv2.UMat(final_watermark_normalized))

 return (finalwatermark)




def steganography(original_image, watermark_image, alpha):

    coverimage = performzeropadding(imageio.imread(original_image)) #read image in RGB
    cover_int=coverimage
    coverimage=coverimage/255
    
    red_cover = coverimage.copy()
    # set green and blue channels to 0
    red_cover[:, :, 1] = 0
    red_cover[:, :, 2] = 0

    green_cover = coverimage.copy()
    # set red and blue channels to 0
    green_cover[:, :, 0] = 0
    green_cover[:, :, 2] = 0

    blue_cover = coverimage.copy()
    # set green and red channels to 0
    blue_cover[:, :, 0] = 0
    blue_cover[:, :, 1] = 0

    #Step-3, Getting DCT for each RGB components

    dct_r_org= cv2.dct(coverimage[:,:,0])
    dct_g_org= cv2.dct(coverimage[:,:,1])
    dct_b_org= cv2.dct(coverimage[:,:,2])
    #checking to see if original figure can be reconstructed using IDCT or not

    DCT_RGB_org = np.zeros((coverimage.shape[0],coverimage.shape[1],3), 'float64')
    DCT_RGB_org[:,:,0] = cv2.idct(dct_r_org)
    DCT_RGB_org[:,:,1] = cv2.idct(dct_g_org)
    DCT_RGB_org[:,:,2] = cv2.idct(dct_b_org)
    normalizedDCT_RGB_org=np.clip(DCT_RGB_org,0,1)
    
    watermarkimage = performzeropadding( imageio.imread(watermark_image)) #read image in RGB
    watermarkimage_int=watermarkimage
    watermarkimage=watermarkimage/255;
    
    #Step-5, Getting DCT for each RGB components for watermark image
    dct_r_w= cv2.dct(watermarkimage[:,:,0])
    dct_g_w= cv2.dct(watermarkimage[:,:,1])
    dct_b_w= cv2.dct(watermarkimage[:,:,2])
    
    #checking to see if original watermark figure can be reconstructed using IDCT or not

    DCT_RGB_wat = np.zeros((watermarkimage.shape[0],watermarkimage.shape[1],3), 'float64')
    DCT_RGB_wat[:,:,0] = cv2.idct(dct_r_w)
    DCT_RGB_wat[:,:,1] = cv2.idct(dct_g_w)
    DCT_RGB_wat[:,:,2] = cv2.idct(dct_b_w)
    normalizedDCT_RGB_wat = np.clip(DCT_RGB_wat,0,1)   # making in range 0 to 1
    
    #step-6, Store the Watermark image into the Cover image  using alpha blending function. alpha=0.99 used here



    dct_r_org[1:watermarkimage.shape[0]+1, 1:watermarkimage.shape[1]+1]= alpha * dct_r_org[1:watermarkimage.shape[0]+1, 1:watermarkimage.shape[1]+1] + (1-alpha) * dct_r_w
    dct_g_org[1:watermarkimage.shape[0]+1, 1:watermarkimage.shape[1]+1]= alpha * dct_g_org[1:watermarkimage.shape[0]+1, 1:watermarkimage.shape[1]+1] + (1-alpha) * dct_g_w
    dct_b_org[1:watermarkimage.shape[0]+1, 1:watermarkimage.shape[1]+1]= alpha * dct_b_org[1:watermarkimage.shape[0]+1, 1:watermarkimage.shape[1]+1] + (1-alpha) * dct_b_w
    
    #Displaying the watermarked image

    idct_r=cv2.idct(dct_r_org)
    idct_g=cv2.idct(dct_g_org)
    idct_b=cv2.idct(dct_b_org)
    stagimage = np.zeros((coverimage.shape[0],coverimage.shape[1],3), 'float64')
    stagimage[:,:,0]=idct_r;
    stagimage[:,:,1]=idct_g;
    stagimage[:,:,2]=idct_b;

    normalizedstagimage=np.clip(stagimage,0,1)
    a3=np.uint8(normalizedstagimage*255)
    rgb_image4 = cv2.cvtColor(a3, cv2.COLOR_BGR2RGB)
    
    # Step 9: PSNR and MSE values are calculated, taking into account the original image and watermarked image.
    # original_image = cv2.imread(original_image)
    psnr = cv2.PSNR(cover_int, rgb_image4)
    mse = ((cover_int - rgb_image4) ** 2).mean()

    # Step 10: Save the watermarked image
    # cv2.imwrite("watermarked_embedtry1.jpg", watermarked_image)


    return rgb_image4, psnr, mse

def extract_watermark(watermarked_image, cover_image, alpha):
    
    #step 1 read watermark image
    watermarkimage = performzeropadding( imageio.imread(watermarked_image)) #read image in RGB
    watermarkimage_int=watermarkimage
    watermarkimage=watermarkimage/255;
    
    #step 2 read cover image
    coverimage = performzeropadding(imageio.imread(cover_image)) #read image in RGB
    cover_int=coverimage
    coverimage=coverimage/255
    
    finalwatermark = extract_watermark1(watermarked_image, coverimage, watermarkimage.shape[0], watermarkimage.shape[1],alpha)
    finalwatermark = np.uint8(finalwatermark)
    final_watermark_normalized = cv2.normalize(finalwatermark, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


    rgb_image6 = cv2.cvtColor(final_watermark_normalized, cv2.COLOR_BGR2RGB)
    # psnr = cv2.PSNR(fn3, rgb_image4)
    # mse = ((fn3 - rgb_image4) ** 2).mean()
    # psnr = cv2.PSNR(np.uint8(watermarked_image*255), watermark)
    # mse = ((np.uint8(watermarked_image*255) - watermark) ** 2).mean()
    psnr =1
    mse = 1
    
    return rgb_image6, psnr , mse
 