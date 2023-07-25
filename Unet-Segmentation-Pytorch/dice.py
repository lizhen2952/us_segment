import numpy as np
from PIL import Image
import  cv2

def iou(a, b, epsilon=1e-5):
    # 首先将a和b按照0/1的方式量化
    a = (a > 0).astype(int)
    b = (b > 0).astype(int)
    
    # 计算交集(intersection)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)
    
    # 计算并集(union)
    union = np.logical_or(a, b)
    union = np.sum(union)
    
    # 计算IoU
    iou = intersection / (union + epsilon)
    
    return iou

def threshold_By_OTSU(input_img_file):
    image=cv2.imread(input_img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   ##要二值化图像，必须先将图像转为灰度图
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("threshold value %s" % ret)  #打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
    # cv2.imshow("threshold", binary) #显示二值化图像
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return binary


# 测试内容
if __name__ == '__main__':
    imgLabel = r'E:\LZ\Unet-Segmentation-Pytorch-Nest-of-Unets-master\unet\pred\mask.png'
    imgPredict = r"E:\\LZ\\1_1.png"
    label_bin=threshold_By_OTSU(imgLabel)
    Predict_bin=threshold_By_OTSU(imgPredict)
    
    
    iou_score=iou(label_bin, Predict_bin)
    print(iou_score)
    
    
    # img_path=r"E:\LZ\Unet-Segmentation-Pytorch-Nest-of-Unets-master\R2AttU_Net\pred\img_iteration_5_epoch_4.png"
    # image=cv2.imread(img_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   ##要二值化图像，必须先将图像转为灰度图
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print("threshold value %s" % ret)  #打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
    # # cv2.imshow("threshold", binary) #显示二值化图像
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # cv2.imwrite("E:\\LZ\\1_1.png",binary)