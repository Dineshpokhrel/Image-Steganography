import cv2
import numpy as np

# Load the watermarked image and the original cover image
# watermarked_image = cv2.imread(r"C:\Users\dines\Desktop\try algorithm\girl.jpg")
# cover_image = cv2.imread(r"C:\Users\dines\Desktop\try algorithm\watermarked_embedtry1.jpg")

def retrive_naya(watermarked_image,cover_image,alpha):
    
    # Split the watermarked image and cover image into RGB components
    b_wm, g_wm, r_wm = cv2.split(watermarked_image)
    b_cover, g_cover, r_cover = cv2.split(cover_image)

    # Perform DCT on each component
    dct_b_wm = cv2.dct(b_wm.astype(float))
    dct_g_wm = cv2.dct(g_wm.astype(float))
    dct_r_wm = cv2.dct(r_wm.astype(float))
    dct_b_cover = cv2.dct(b_cover.astype(float))
    dct_g_cover = cv2.dct(g_cover.astype(float))
    dct_r_cover = cv2.dct(r_cover.astype(float))

    # Extract the watermark using the given alpha value
    alpha = 0.95
    # b_wm_extracted = ((dct_b_wm - dct_b_cover) * alpha) / (1 - alpha)
    # g_wm_extracted = ((dct_g_wm - dct_g_cover) * alpha) / (1 - alpha)
    # r_wm_extracted = ((dct_r_wm - dct_r_cover) * alpha) / (1 - alpha)

    b_wm_extracted = ((dct_b_wm - dct_b_cover) / alpha)
    g_wm_extracted = ((dct_g_wm - dct_g_cover) / alpha)
    r_wm_extracted = ((dct_r_wm - dct_r_cover) / alpha)

    # Perform IDCT on each component to get the watermark image
    watermark_b = cv2.idct(b_wm_extracted).astype('uint8')
    watermark_g = cv2.idct(g_wm_extracted).astype('uint8')
    watermark_r = cv2.idct(r_wm_extracted).astype('uint8')

    # Average the RGB components to get the final watermark image
    watermark = cv2.merge((watermark_b, watermark_g, watermark_r))
    watermark = np.mean(watermark, axis=2)

    # Save the extracted watermark image
    # cv2.imwrite("extracted_watermark_try1.jpg", watermark)
    return watermark
