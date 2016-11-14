#coding:utf-8
import cv2
import os
import numpy as np
import random

AUGMENTATION_TIMES = 1

file_path = "/media/onegene/本地磁盘/数据集/IHDPHeadPose/"

new_file_path = "/home/onegene/file/Face/IHDPHeadPose/"

angle_shift = 10
position_shift = 5

def flip_right_left(fo):
    label_type = fo.split('_')
    if len(label_type) > 1:
        if label_type[1] == 'left':
            return label_type[0]+'_right'
        else:
            return label_type[0]+'_left'
    else:
        return fo

def bgr_jitter(img):
    h, w, c = img.shape
    noise = np.random.randint(0,20,(h, w)) # design jitter/noise here
    jitter = np.zeros_like(img)
    random_channel  = np.random.randint(0, 2)
    jitter[:,:,random_channel] = noise
    return cv2.add(img, jitter)

# argement datas with angle in (-10, 10) and certer position shift in (-5, 5)
def augmentation(img):
    top_x = np.random.randint(0, 10)
    top_y = np.random.randint(0, 10)
    img_roi = img[top_y: top_y+64, top_x: top_x+64]
    new_img_roi = cv2.resize(img_roi,(32,32))
    (h, w) = new_img_roi.shape[:2]
    center = (w / 2, h / 2)
    ang = np.random.randint(-angle_shift, angle_shift)
    #print("ang: "+str(ang))
    M = cv2.getRotationMatrix2D(center, ang, 1.0)
    rotated = cv2.warpAffine(new_img_roi, M, (w, h))
    return rotated

def pan2label(pan):
    if pan < -0.6:
        return 'profile_left'
    if pan >= -0.6 and pan < 0:
        return 'frontal_left'
    if pan >= 0 and pan < 0.6:
        return 'frontal_right'
    if pan >= 0.6:
        return 'profile_right'
def storeRoi(img_roi, pan, pic_name, isTestSet=True):
    face_orient = pan2label(pan)    
    pic_name = pic_name.split('.')[0]
    print(pic_name)
    
    for i in range(0, AUGMENTATION_TIMES):
        if isTestSet:
            img_roi_flipped = cv2.flip(img_roi, 1)
            img_roi_flipped = bgr_jitter(img_roi_flipped)
            face_orient_flip = flip_right_left(face_orient)
            img_roi_flipped_augmentation = augmentation(img_roi_flipped)
            if not os.path.exists(new_file_path+frame_floder):
                os.makedirs(new_file_path+frame_floder)
            cv2.imwrite(new_file_path+frame_floder+pic_name+'~f~'+str(i)+'-'+face_orient_flip+'.jpg',img_roi_flipped_augmentation)
        
        if isTestSet == False:
            pic_name_splited = pic_name.split('_')
            if len(pic_name_splited) > 1:
                pic_name = pic_name_splited[0]+'~f'
                img_roi = bgr_jitter(img_roi)
        img_roi_augmentation = augmentation(img_roi)
        if not os.path.exists(new_file_path+frame_floder):
                os.makedirs(new_file_path+frame_floder)
        cv2.imwrite(new_file_path+frame_floder+pic_name+'~'+str(i)+'-'+face_orient+'.jpg',img_roi_augmentation)
        

print("Process start!\n")
print(" start")
frame_floder = "test/"
annot_file = "test_anno.txt"
annot = open(file_path+annot_file, 'r')
num_frames = int(annot.readline())
print('There are '+str(num_frames)+' to process.')

try:
    for frame in range(0, num_frames):
        peer = annot.readline().split()
        pic_name = peer[0]
        # If pan > 0.6 or pan < -0.6, we consider it as profile(17448 out of 42304), otherwise it's frontal
        pan =  float(peer[1])
        img = cv2.imread(file_path+frame_floder+pic_name)
        storeRoi(img, pan, pic_name)

finally:
    annot.close()
    cv2.destroyAllWindows()

print("\nProcess end!")
