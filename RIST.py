import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

import radius_match_essential
import create_map
import tracking
import os

class RIST():

    __query_imgs = []
    __max_t = [0, 0, 0]
    __db_img_answer = []

    __answer_translation = [0, 0]
    __answer_translation_temp = [0, 0]
    __max_answer_translation = [0, 0]

    __rm_max_match_length = 0


    # 초기화
    def __init__(self) -> None:
        
        self.__part = int(input("Part : "))
        self.__answer = int(input("쿼리 영상의 인덱스를 입력하시오 : "))
        self.__answer_range = float(input("Range : "))

        self.track = tracking.Tracking()
        self.rm = radius_match_essential.Radius_Match()
        self.cm = create_map.DB_MAP()  
        
        self.import_val() # yaml 파일로부터 변수를 불러오는 함수
        self.import_img() # path에서 이미지 불러오는 함수
    
    # 찾고자 하는 q 순서 반환
    def get_answer(self) -> int:
        return self.__answer
    
    # yaml 파일로부터 변수를 불러오는 함수
    def import_val(self) -> None: 
        with open('part' + str(self.__part) + '.yaml', 'r') as f:
            self.__data = yaml.safe_load(f)

        self.__rmvoutlier_th = self.__data['rmvoutlier_th']
        self.__angle_th = self.__data['angle_th']
        self.__num_images = self.__data['num_images']
        

        # Create MAP
        self.cm.scale_x = self.__data['scale_x']
        self.cm.scale_y = self.__data['scale_y']
        self.cm.db_path = self.__data['db_path']

        self.__db_path = self.cm.db_path
        self.__q_path = self.__data['q_path']
        # self.__num_images = self.get_num_images(self.__q_path)
        
        self.__undistortion = self.__data['undistortion'] 
        self.__scat_all_db = self.__data['scat_all_db']
        self.__scat_max_db = self.__data['scat_max_db']

        # Matching
        self.rm.radius = self.__data['radius']
        self.rm.K_d = np.array([[1.88 , 0.0  , 1376.31],
                                [0.0  , 1.88 , 1375.22],
                                [0.0  , 0.0  , 1.0]])
        self.rm.draw_circle = self.__data['draw_circle']
        self.rm.radiusMatching = self.__data['radiusMatching']
        self.rm.nfeatures = self.__data['nfeatures']

    # 파일 경로로부터 이미지 불러와서 저장
    def import_img(self) -> None:
        if(self.__undistortion == 1):
            """undistort"""
            for i in range(0, self.__num_images):
                img = cv2.imread(self.__q_path + 'undistort_' + str(i) + ".jpg")
                self.__query_imgs.append(img)
            
            self.__query_size = (self.__query_imgs[0].shape[0], self.__query_imgs[0].shape[1])
            
            # undistortion images camera intrinsic parameter
            self.rm.K = np.array([[301.39596558 , 0.0          , 316.70672662],
                        [0.0          , 300.95941162 , 251.54445701],
                        [0.0          , 0.0          , 1.0]])
            
        else:
            """distort"""
            for i in range(0, self.__num_images):
                img = cv2.imread("/home/aaron/RIST/dataset/query_distort/" + str(i) + ".jpg")
                self.__query_imgs.append(img)
                
            self.__query_size = (self.__query_imgs[0].shape[0], self.__query_imgs[0].shape[1])

            # distortion images camera intrinsic parameter
            self.rm.K = np.array([[301.867624408757 , 0.0                , 317.20235900477695],
                                [0.0              , 301.58768437338944 , 252.0695806789168],
                                [0.0              , 0.0                , 1.0]])


    def tracking(self, i, focal_, pp_) -> None:

        img1 = self.__query_imgs[i-1]
        img2 = self.__query_imgs[i]

        self.track.calc_optical_flow(img1, img2)
        
        self.track.remove_outliers(self.__rmvoutlier_th)
        
        self.track.get_mean_direction_vector()
        
        self.track.get_angle()
        
        self.track.get_Rt(focal_, pp_)

        self.track.detect_rotation(self.__angle_th)
    
        self.__translation_xy = self.track.get_translation()


    def matching(self , i) -> None:

        # #MATCHING ANSWER_QUERY WITH DB
        self.__db_search_index = self.cm.db_matching(self.__translation_xy , self.__answer_range)

        self.cm.Scatter(self.__translation_xy[0] , self.__translation_xy[1], 180, 'red')


        if(self.track.get_flags_val() == 0):
            T_flags = 1
        elif(self.track.get_flags_val() == 1):
            T_flags = 0

        self.rm.flags = T_flags

        for j in self.__db_search_index:
            # db_img = cv2.imread(db_path + str(db_imgs[j]))
            db_img = cv2.imread(self.__db_path + str(self.cm.get_db_imgs()[j]))
            db_img = cv2.resize(db_img, (self.__query_size[1], self.__query_size[0]))

            new_img_matches, new_matches, rm_R, rm_t = self.rm.radius_match(self.__query_imgs[i], db_img)
            
            if self.__rm_max_match_length < len(new_matches):
                self.__rm_max_match_length = len(new_matches)
                self.__max_t[0] = rm_t[0]
                self.__max_t[1] = rm_t[1]
                self.__max_t[2] = rm_t[2]
                self.__db_img_answer.append(db_img)

            self.rm.t = self.__max_t
                
            self.__answer_translation = self.__max_answer_translation.copy()
            
            # answer_translation_temp[0] = answer_translation[0] + rm_t[0]
            # answer_translation_temp[1] = answer_translation[1] + rm_t[1]
            
            self.__answer_translation_temp = self.rm.get_translation(rm_t, self.__answer_translation, T_flags, self.track.get_direction_val())
            
            ####################################################################################################
            if(self.__scat_all_db == 1):
                self.cm.Scatter(self.__answer_translation_temp[0], self.__answer_translation_temp[1], 90, 'green')
                self.cm.Annotate(f"{i}Q - {j}D", (self.__answer_translation_temp[0], self.__answer_translation_temp[1]), 6)
            ####################################################################################################

        if i == self.__answer: 
            result = cv2.hconcat([self.__query_imgs[i], self.__db_img_answer[-1]])

            plt.subplots()
            plt.imshow(result)
        
        self.__max_answer_translation = self.rm.get_translation(self.__max_t , self.__max_answer_translation, T_flags, self.track.get_direction_val())
        ####################################################################################################
        if(self.__scat_max_db == 1):
            self.cm.Scatter(self.__max_answer_translation[0], self.__max_answer_translation[1], 50, 'yellow')
        ####################################################################################################
    

    
    # 경로 안에 이미지가 몇 개 있는지 출력하는 함수
    def get_num_images(self, q_path) -> int:

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