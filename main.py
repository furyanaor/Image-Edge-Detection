import cv2
import numpy as np
from lab4.myMasking import my_masking as mm
from MyFunctions.showimg import showimg as sm
from MyFunctions.test import test as tes


org_img = cv2.imread(r"c:\\ImgC.jpg")
Gaussian_Filter_Mask = (1/273.0) * np.array([[1, 4, 7, 4, 1],
                                             [4, 16, 26, 16, 4],
                                             [7, 26, 41, 26, 7],
                                             [4, 16, 26, 16, 4],
                                             [1, 4, 7, 4, 1]], dtype=np.float64)
Sobel_Gradient_Mask_X = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float64)
Sobel_Gradient_Mask_Y = np.array([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]], dtype=np.float64)

gauss_img = mm(org_img, Gaussian_Filter_Mask)
new_img_x = mm(gauss_img, Sobel_Gradient_Mask_X)
new_img_y = mm(gauss_img, Sobel_Gradient_Mask_Y)

# gauss_img = mm(np.float64(org_img), Gaussian_Filter_Mask)
# new_img_x = mm(np.float64(gauss_img), Sobel_Gradient_Mask_X)
# new_img_y = mm(np.float64(gauss_img), Sobel_Gradient_Mask_Y)

new_img = np.sqrt(np.float64(new_img_x) ** 2 + np.float64(new_img_y) ** 2)
new_img -= np.min(new_img)
new_img *= (255 / np.max(new_img))
new_img = np.uint8(new_img)
# new_img = tes(org_img)

sm(org_img, new_img)
