import pytesseract
import shutil
import os
import glob
import cv2
import re
import numpy as np
import tqdm
import matplotlib.pyplot as plt

def ocr_data(image):
    """
    :param image:  ROI for the txt bounding box
    :return: [d1,d2]
    """
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, 127,255, cv2.THRESH_BINARY_INV)
    # image=cv2.medianBlur(image,3)            # do not use this for 7 to ?
    txt=pytesseract.image_to_string(image)
    # print(txt)
    data=re.findall(r'\d+\.\d+',txt)
    # plt.imshow(image, cmap='gray')
    # plt.show()
    if len(data) < 2 or '?'in txt:
        print(txt)
        plt.imshow(image, cmap='gray')
        plt.show()
    return data
def detect_boundbox(image,filename=None,save_txt_fig=True,save_roi=True,save_result=True):
    """
    :param image:  origina image
    :return: [box]
    """
    sr_image=image.copy()
    color_range=[([0,200,200],[100,255,255])] # RGB
    for (lower,uper) in color_range:
        lower=np.array(lower,dtype="uint8")
        upper=np.array(uper,dtype="uint8")
        mask=cv2.inRange(image,lower,upper)
        output=cv2.bitwise_and(image,image,mask=mask)
    image=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)           # RGB2Gray
    image= cv2.GaussianBlur(image, (9, 9), 0)               # blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100)) #morphology
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(image, None, iterations=4)           #erase & dilate
    image= cv2.dilate(closed, None, iterations=4)
    cnts,_=cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)
    boxes=[cv2.boundingRect(cnts[x]) for x in range(2)] #(x,y,w,h)
    boxes=sorted(boxes,key=lambda x:x[0])               # sort the box
    x,y,w,h=boxes[0]                                    # lesions
    center_x=y+h//2
    center_y=x+w//2
    txt_x,txt_y,txt_w,txt_h=boxes[1]                     # txt box
    if save_roi:
        cv2.imwrite(os.path.join('roi',filename),cv2.cvtColor(sr_image[y:y+h,x:x+w,:],cv2.COLOR_RGB2BGR))
    roi=sr_image[txt_y-2:txt_y+txt_h+2,txt_x-2:txt_x+txt_w+2,:]     # add some offset
    if save_txt_fig:
        cv2.imwrite(os.path.join('save',filename),cv2.cvtColor(roi,cv2.COLOR_RGB2BGR))                 # save txt file
    # commend for the "?" problem
    # roi=cv2.resize(roi,(2*(txt_w+2),2*(txt_h+2)))                   # rescale for better recognition , keep aspect ratio

    data=ocr_data(roi)
    if len(data)<2:
        print("can't finish fig:{}. Image size {}x{}".format(filename,txt_w,txt_h))
    elif not data:
        print("can't recognize fig:{}. Image size {}x{}".format(filename,txt_w,txt_h))
    elif len(data[0].split('.')[1])>2 or len(data[1].split('.')[1])>2:
        print('there maybe some error')
        print(data)
    if save_result:
        # save new_file with information in the filename
        name=os.path.splitext(filename)[0]
        infor="x_{}_y_{}_D1_{}_D2_{}".format(center_x,center_y,data[0],data[1])
        extend=os.path.splitext(filename)[-1]
        shutil.copy(filename,os.path.join('result',name+infor+extend))

    # plt.imshow(roi,cmap=plt.cm.gray)
    # plt.show()

def process(files):
    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detect_boundbox(image, file)


if __name__ == '__main__':
    files=glob.glob("*.png")
    files=sorted(files)
    print("process file {}".format(len(files)))
    process(files)

