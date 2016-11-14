#coding:utf-8
import cv2
import os
import numpy as np
import Image
import random
file_path = "/media/onegene/本地磁盘/数据集/ut/"

annot_floder = "interclips_anno/"

frm_floder = "frms_inter/"

new_file_path = "/home/onegene/file/Face/ut/"

AUGMENTATION_TIMES = 3

angle_shift = 10
position_shift = 2

# argement datas with angle in (-10, 10) and certer position shift in (-5, 5)
def augmentation(img):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    ang = random.randint(-angle_shift, angle_shift)
    pos = random.randint(-position_shift, position_shift)
    M = cv2.getRotationMatrix2D(center, ang, 1.0)
    rotated = cv2.warpAffine(img, M, (w+pos, h+pos))
    return rotated

def storeRoi(img, tlx, head_left, tly, brx, head_right, bry, fo):
    if head_right-head_left > 60 or head_right-head_left < 25:
        return
    new_tlx = tlx+head_left
    new_brx = tlx+head_right
    if new_tlx<0:
        new_tlx = 0
    if new_brx<0:
        new_brx = 0
    #To get a better (smaller) bounding box of the face
    box_expend = int(0.1*(head_left-head_right))
    tlx_sub_box_expend = new_tlx - box_expend
    tly_sub_box_expend = tly - box_expend
    brx_add_box_expend = new_brx + box_expend
    bry_add_box_expend = bry + box_expend
    if tlx_sub_box_expend<0:
        tlx_sub_box_expend = 0
    if tly_sub_box_expend<0:
        tly_sub_box_expend = 0
    img_roi = img[tly_sub_box_expend: bry_add_box_expend, tlx_sub_box_expend: brx_add_box_expend]
    for i in range(0, AUGMENTATION_TIMES):
        img_augmentation = augmentation(img_roi)
        new_img_roi = cv2.resize(img_augmentation,(32,32))
        if not os.path.exists(new_file_path+frame_floder):
            os.makedirs(new_file_path+frame_floder)
        cv2.imwrite(new_file_path+frame_floder+pic_name+'-'+str(box)+'~'+str(i)+'-'+fo+'.jpg',new_img_roi)

print("Process start!\n")
for type_id in os.listdir(file_path+frm_floder):
    print(type_id+" start")
    frame_floder = frm_floder+type_id+"/"
    annot_file = type_id+".anno"
    annot = open(file_path+annot_floder+annot_file, "r")

    try:
        while 1:
            first_readed_line = annot.readline()
            if not first_readed_line:
                break
            frame_first_line = first_readed_line.split()
            
            frame_num = int(frame_first_line[1])
            pic_name = str(frame_num).zfill(3)
            img = cv2.imread(file_path+frame_floder+pic_name+'.jpg')
            num_bbxs = int(frame_first_line[3])
            for box in range(0, num_bbxs):
                peer = annot.readline().split()
                top_left_x = int(peer[1])
                top_left_y = int(peer[2])
                botton_right_x = int(peer[3])
                botton_right_y = int(peer[4])
                face_orient = peer[6]
                new_bry = int(0.2*botton_right_y+0.8*top_left_y)
                
                img_roi = img[top_left_y: new_bry, top_left_x: botton_right_x]
                m = []
                for i in range(0, img_roi.shape[1]):
                    mean_light = np.mean(img_roi[0:int(img_roi.shape[1]*1), i])
                    m.append((mean_light))
                m_arr = np.asarray(m)
                m_min = np.argmin(m_arr)
                diff = np.asarray(m[1: m_arr.size])-np.asarray(m[0:m_arr.size-1])
                mean_3_diff = (diff[0: diff.size-2]+diff[1:diff.size-1]+diff[2:diff.size])/3
                m3f_min = np.argmin(mean_3_diff)
                m3f_max = np.argmax(mean_3_diff)
                #Two parameters below are the left and right bounds of head
                pic_m3f_min = m3f_min-4
                pic_m3f_max = m3f_max+6
                if pic_m3f_min < 0:
                    pic_m3f_min = 0
                if pic_m3f_max > img_roi.shape[1]-1:
                    pic_m3f_max = img_roi.shape[1]-1
                storeRoi(img, top_left_x, pic_m3f_min, top_left_y, botton_right_x, pic_m3f_max, new_bry, face_orient)

    finally:
        annot.close()
        #new_anno_file.close()
        cv2.destroyAllWindows()
        print(type_id+" finish\n")
        
print("\nProcess end!")
