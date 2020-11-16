import cv2
import matplotlib.pyplot as plt 



img=cv2.imread("./data/images/Img_RSKA003603_0_r2944_c896.png")
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
bb_file = open("../labels/Img_RSKA003603_0_r2944_c896.txt")
lines = bb_file.readlines()
content = [x.strip() for x in lines]
for line in content:
    bbox = line.split(" ")
    bbox = [int(i) for i in bbox]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),color=(0, 255, 0), thickness=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    # rgb1= np.rollaxis(rgb1, 0,3)  
ax[0].imshow(img)
ax[0].set_title('image')
# ax[1].imshow(masks[1280:1280 + 256,3072:3072 + 256], cmap='gray')
# ax[1].set_title('Masks 1')
# plt.imshow(img)
plt.show()