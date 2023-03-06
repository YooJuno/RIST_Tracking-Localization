import matplotlib.pyplot as plt
import time
import mainclass 

if __name__ == '__main__':

    # main 객체 생성
    rist = mainclass.RIST()

    # 이미지 뽑아놓기
    rist.extract_img()

    #MAKE DB MAP
    rist.cm.db_map()

    ############## Main Loop ##############
    for i in range(1, rist.get_answer()+1):

        t_start = time.time()

        #################### Tracking Part ###################
        rist.tracking( i, 301, (316. , 251.))
        
        #################### Matching Part ####################
        rist.matching(i)

        t_end = time.time()
        print(f"{t_end - t_start:.5f} sec")

        plt.pause(0.1)

    plt.show()