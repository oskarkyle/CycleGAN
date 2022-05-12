import cv2
import numpy as np
def preprocess(img, imgsize, jitter, random_placing=False):
    """
    Image preprocess for yolo input
    Pad the shorter side of the image and resize to (imgsize, imgsize)
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        imgsize (int): target image size after pre-processing
        jitter (float): amplitude of jitter for resizing
        random_placing (bool): if True, place the image at random position
    Returns:
        img (numpy.ndarray): input image whose shape is :math:`(C, imgsize, imgsize)`.
            Values range from 0 to 1.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    """
    h, w, _ = img.shape
    img = img[:, :, ::-1]
    assert img is not None
    #尺寸大小的随机抖动，jitter越大，长宽的的变化越大
    if jitter > 0:
        # add jitter
        dw = jitter * w
        dh = jitter * h
        new_ar = (w + np.random.uniform(low=-dw, high=dw))\
                 / (h + np.random.uniform(low=-dh, high=dh))
    else:
        new_ar = w / h
 
    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)
    #图像填充位置的随机性
    if random_placing:
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2
 
    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy:dy+nh, dx:dx+nw, :] = img
 
    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img
 
jitter=0.9
andom_placing=False
img_size=512
img=cv2.imread('test_img-1.jpg')
print(img.shape)
sized, info_img=preprocess(img, img_size, jitter=jitter,random_placing=andom_placing)
print(sized.shape)
sized=sized[:,:,::-1]
cv2.imshow('imgs',img)
cv2.imshow('img',sized)
cv2.waitKey()
 