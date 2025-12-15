"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import time

videos = glob.glob('C:/Users/Lenovo/Desktop/codes/matlab/box_0425_200-1000.avi')
if not videos:
        print("no video founded")
for video in videos:
    video_path = os.path.abspath(video)
    capture = cv2.VideoCapture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    frames = []
    bin_size = 1
    spatial_decompose = 2 #空间降维参数,越大计算越快
    #std_per5 = []
    
    """ 
"""
    ret, prev_frame = capture.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(prev_gray)
    frame_count +=1
    """
"""
    while True:
        ret, curr_frame = capture.read()
        if not ret:
            break
       
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        #curr_gray = cv2.GaussianBlur(curr_gray, (15, 15), 0)
        frame_count +=1

        ccls = cv2.HoughCircles(
            curr_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,           # 累加器分辨率与图像分辨率的反比
            minDist=30,     # 检测到的圆心之间的最小距离
            param1=50,      # Canny边缘检测的高阈值
            param2=30,      # 圆心检测阈值
            minRadius=15,    # 最小半径
            maxRadius=30  # 最大半径
        )
        if ccls is not None:
            ccls = np.round(ccls[0, :]).astype("int")
            for (x, y, r) in ccls:
                cv2.circle(curr_gray, (x, y), r, (0, 255, 0), 2)  
                cv2.circle(curr_gray, (x, y), 2, (0, 0, 255), 3)  
        else:
            print(f" {frame_count}:not found")

        cv2.imshow('circles on raw frames',curr_gray)
        cv2.waitKey(50)  

        if frame_count % 50 == 0:
            print(f"{frame_count}/{total_frames}")
            
    capture.release()
    cv2.destroyAllWindows()

    video_stack = np.stack(frames,axis=2)
    print(f"a stack of {frame_count} frames has been formed")
    print(f"shape = {video_stack.shape}")
    std_image = np.std(video_stack,axis=2)
    
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import time

videos = glob.glob('C:/Users/Lenovo/Desktop/codes/matlab/box_0425_200-1000.avi')
if not videos:
        print("no video founded")
for video in videos:
    video_path = os.path.abspath(video)
    capture = cv2.VideoCapture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    frames = []
    bin_size = 1
    spatial_decompose = 1 #空间降维参数,越大计算越快
    
    # 创建VideoWriter来保存结果视频
    output_fps = fps / bin_size if fps / bin_size > 0 else 10
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 计算缩小后的尺寸
    output_width = int(width/spatial_decompose)
    output_height = int(height/spatial_decompose)
    
    # 创建两个VideoWriter
    out_std = cv2.VideoWriter('raw_std_output.avi', fourcc, output_fps, (output_width, output_height), isColor=False)
    out_circles = cv2.VideoWriter('raw_circles_output.avi', fourcc, output_fps, (output_width, output_height), isColor=True)
    tic = time.time()
    while True:
        ret, curr_frame = capture.read()
        if not ret:
            break
       
        # 处理原始彩色帧用于圆圈检测
        curr_frame_resized = cv2.resize(curr_frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        curr_gray = cv2.cvtColor(curr_frame_resized, cv2.COLOR_BGR2GRAY)
        
        frame_count += 1

        # 在灰度图像上检测圆圈
        ccls = cv2.HoughCircles(
            curr_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,           # 累加器分辨率与图像分辨率的反比
            minDist=30,     # 检测到的圆心之间的最小距离
            param1=50,      # Canny边缘检测的高阈值
            param2=30,      # 圆心检测阈值
            minRadius=10,    # 最小半径
            maxRadius=20  # 最大半径
        )
        
        # 创建彩色图像用于显示圆圈结果
        circles_frame = curr_frame_resized.copy()
        
        if ccls is not None:
            ccls = np.round(ccls[0, :]).astype("int")
            for (x, y, r) in ccls:
                # 在彩色图像上绘制彩色圆圈
                cv2.circle(circles_frame, (x, y), r, (0, 255, 0), 2)  # 绿色圆圈
                cv2.circle(circles_frame, (x, y), 2, (0, 0, 255), 3)  # 红色圆心
            print(f"{frame_count}: found {len(ccls)} circles")
        else:
            print(f"{frame_count}: not found")
        
        # 保存到视频文件
        out_std.write(curr_gray)           # 保存灰度图像
        out_circles.write(circles_frame)   # 保存带彩色圆圈的图像
        
        # 显示结果
        cv2.imshow('Circles Detection', circles_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        if frame_count % 50 == 0:
            print(f"Progress: {frame_count}/{total_frames}")
    toc = time.time()          
    print(f"执行时间: {toc - tic:.4f} 秒")
    capture.release()
    out_std.release()
    out_circles.release()
    cv2.destroyAllWindows()
"""
    
    
    for i in range(len(std_per5)):
        roi = std_image * std_per5[i]
        roi_normalized = (roi - roi.min()) / (roi.max() - roi.min()) * 255
        roi_display = roi_normalized.astype(np.uint8)
        result_display = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
            
        circles = cv2.HoughCircles(
            roi_display,
            cv2.HOUGH_GRADIENT,
            dp=1,           # 累加器分辨率与图像分辨率的反比
            minDist=25,     # 检测到的圆心之间的最小距离
            param1=50,      # Canny边缘检测的高阈值
            param2=30,      # 圆心检测阈值
            minRadius=20,    # 最小半径
            maxRadius=30 # 最大半径
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(result_display, (x, y), r, (0, 255, 0), 2)  
                cv2.circle(result_display, (x, y), 2, (0, 0, 255), 3)  
        else:
            print(f" {frame_count}:not found")
        
        cv2.imshow('std of videoframes', roi_display)
        cv2.imshow('circles of videoframes', result_display)
        cv2.waitKey(50)  
"""



"""
    threshold = 10
    ROI = std_image > threshold
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(std_image,cmap='hot')
    plt.title('std of videoframes')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(ROI, cmap='gray')
    plt.title('roi of std')
    
    plt.tight_layout()
    plt.show()
    time.sleep(0.1)
    plt.clf() 
    
"""

"""
    std_normalized = (std_image - std_image.min()) / (std_image.max() - std_image.min()) * 255
    cv2.imshow('std of videoframes', std_normalized.astype(np.uint8))
    cv2.imshow('roi of std', ROI.astype(np.uint8)*255)
    cv2.waitKey(50)  
    
    
    
    
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        std_diff = np.std(frame_diff)
        if len(frame_diff > std_diff)> :
             cv2.imshow('diff',frame_diff)
             cv2.waitKey(50)
        prev_gray = curr_gray
"""

 # seperate = []
    # for i in range(len(seperate)):
    #     stack1 = video_stack[:, :, :m]    # 前m帧
    #     stack2 = video_stack[:, :, m:]    # 第m帧之后的所有帧
