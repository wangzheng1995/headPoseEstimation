#coding:utf-8
import cv2
import os
import numpy as np
import random

AUGMENTATION_TIMES = 2

file_path = "/media/onegene/本地磁盘/数据集/highfive/"

annot_floder = "tv_human_interaction_annotations/"

new_file_path = "/home/onegene/file/Face/highfive/"

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
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    ang = np.random.randint(-angle_shift, angle_shift)
    pos = np.random.randint(-position_shift, position_shift)
    M = cv2.getRotationMatrix2D(center, ang, 1.0)
    rotated = cv2.warpAffine(img, M, (w+pos, h+pos))
    return rotated

def storeRoi(img, tlx, tly, bs, fo, frame_num, pic_name):
    #To get a better (smaller) bounding box of the face
    box_expend = int(0.05*bs)
    tlx_sub_box_expend = tlx+int(bs*0.2)-box_expend
    tly_sub_box_expend = tly-box_expend
    brx_add_box_expend = tlx+int(bs*0.8)+1+box_expend
    bry_add_box_expend = tly+int(bs*0.65)+1+box_expend
    if tlx_sub_box_expend<0:
        tlx_sub_box_expend = 0
    if tly_sub_box_expend<0:
        tly_sub_box_expend = 0
    img_roi = img[tly_sub_box_expend: bry_add_box_expend, tlx_sub_box_expend: brx_add_box_expend]
    img_roi_flipped = cv2.flip(img_roi, 1)
    img_roi_flipped = bgr_jitter(img_roi_flipped)
    fo_flip = flip_right_left(fo)
    for i in range(0, AUGMENTATION_TIMES):
        img_roi_augmentation = augmentation(img_roi)
        img_roi_flipped_augmentation = augmentation(img_roi_flipped)
        new_img_roi = cv2.resize(img_roi_augmentation,(32,32))
        new_img_roi_flipped = cv2.resize(img_roi_flipped_augmentation,(32,32))
        if not os.path.exists(new_file_path+frame_floder):
            os.makedirs(new_file_path+frame_floder)
        cv2.imwrite(new_file_path+frame_floder+pic_name+'-'+str(box)+'~'+str(i)+'-'+fo+'.jpg',new_img_roi)
        cv2.imwrite(new_file_path+frame_floder+pic_name+'-'+str(box)+'~f~'+str(i)+'-'+fo_flip+'.jpg',new_img_roi_flipped)

print("Process start!\n")
for type_id in os.listdir(file_path+"frm"):
    print(type_id+" start")
    frame_floder = "frm/"+type_id+"/"
    annot_file = type_id+".annotations"
    annot = open(file_path+annot_floder+annot_file, "r")
    first_line = annot.readline()
    first_line_split = first_line.split() 
    num_frames = int(first_line_split[1])
    print(str(num_frames)+" frames to deal with")

    new_anno_file = open(new_file_path+annot_floder+annot_file, "w")
    new_anno_file.write(first_line)
    new_anno_file.write("picsize: 32\n")


    try:
        for frame in range(0, num_frames):
            frame_first_line = annot.readline().split()
            frame_num = int(frame_first_line[1])
            pic_name = str(frame_num+1).zfill(4)
            img = cv2.imread(file_path+frame_floder+pic_name+'.jpg')
            num_bbxs = int(frame_first_line[3])
            new_anno_file.write("frame: "+frame_first_line[1]+" num_bbxs: "+frame_first_line[3]+"\n")
            for box in range(0, num_bbxs):
                peer = annot.readline().split()
                top_left_x = int(peer[1])
                top_left_y = int(peer[2])
                box_size = int(peer[3])
                face_orient = peer[5]
                storeRoi(img, top_left_x, top_left_y, box_size, face_orient, frame_num, pic_name)
                new_anno_file.write(face_orient+'\n')
        if num_frames != frame_num+1:
            print("\t\t"+str(num_frames)+" != "+str(frame_num+1))
    finally:
        annot.close()
        new_anno_file.close()
        cv2.destroyAllWindows()
        print(type_id+" finish\n")
        
print("\nProcess end!")
