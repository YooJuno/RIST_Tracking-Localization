import cv2
import numpy as np
import math
import os
import natsort
import matplotlib.pyplot as plt



def qimg_append(num_images_, q_path_,  undistortion_):

    query_imgs_ = []

    if(undistortion_ == 1):
        """undistort"""
        for i in range(0, num_images_):
            img_ = cv2.imread(q_path_+'undistort_' + str(i) + ".jpg")
            query_imgs_.append(img_)
        
        query_size_ = (query_imgs_[0].shape[0], query_imgs_[0].shape[1])
        
        # undistortion images camera intrinsic parameter
        K_ = np.array([[301.39596558, 0.0, 316.70672662],
                                [0.0, 300.95941162, 251.54445701],
                                [0.0, 0.0, 1.0]])
        
    else:
        """distort"""
        for i in range(0, num_images_):
            img_ = cv2.imread("/home/aaron/RIST/dataset/query_distort/" + str(i) + ".jpg")
            query_imgs_.append(img_)
            
        query_size_ = (query_imgs_[0].shape[0], query_imgs_[0].shape[1])

        # distortion images camera intrinsic parameter
        K_ = np.array([[301.867624408757, 0.0, 317.20235900477695],
                                [0.0, 301.58768437338944, 252.0695806789168],
                                [0.0, 0.0, 1.0]])

    return query_imgs_ , K_ , query_size_


def get_num_images(q_path):

    valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]  # 이미지 파일 확장자

    # 폴더 내의 파일 목록 가져오기
    file_list = os.listdir(q_path)

    # 이미지 파일 개수 초기화
    num_images_ = 0

    # 폴더 내의 파일 목록 순회하며 이미지 파일 개수 카운트
    for file_name_ in file_list:
        ext = os.path.splitext(file_name_)[-1].lower()  # 파일 확장자 추출
        if ext in valid_extensions:
            num_images_ += 1
    
    num_images_ -= 1
    return num_images_