import cv2

imagePath='demo2.JPEG'
grid=[7,7]  #width height






img = cv2.imread(imagePath)
h,w,c=img.shape
grid_w = w/grid[0]
grid_h = h/grid[1]
for i in range(1,grid[0]):
    cv2.line(img, (int(i*grid_w),0), (int(i*grid_w),h), (0, 0, 0), 1, 1)

for j in range(1,grid[1]):
    cv2.line(img, (0, int(j*grid_h)), (w, int(j*grid_h)), (0, 0, 0), 1, 1)

# cv2.line(img, (10, 100), (100, 100), (255, 255, 255), 1, 1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
