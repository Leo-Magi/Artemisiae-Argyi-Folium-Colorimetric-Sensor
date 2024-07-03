import cv2
import numpy as np

img = cv2.imread('')

write_name = ''

target_gray =  [0.93, 0.93, 0.93]  # 浮点数范围为0到1

ref_point = None

def mouse_callback(event, x, y, flags, param):
    global ref_point
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = (y, x)

gamma = 2

img = np.power(img / 255.0, gamma) * 255.0
img = np.clip(img, 0, 255).astype(np.uint8)

img = img.astype(np.float32) / 255.0

cv2.namedWindow('Select Reference Point')
cv2.setMouseCallback('Select Reference Point', mouse_callback)
while True:
    cv2.imshow('Select Reference Point', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or ref_point is not None:
        break

if ref_point is not None:
    row, col = ref_point
    ref_pixel = img[row, col]

    img_adjusted = img * (target_gray / ref_pixel)
else:
    img_adjusted = img.copy()

img_adjusted = np.clip(img_adjusted * 255, 0, 255).astype(np.uint8)

cv2.imshow('Original Image', img)
cv2.imshow('Gamma Corrected Image', img)
cv2.imshow('Adjusted Image', img_adjusted)
cv2.imwrite(write_name, img_adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()