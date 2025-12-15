"""
STD-based fiber localization pipeline -- suitable for non-rigid shifts.

This script implements the finalized std-based method
used for our automatic fiber localization and calibration.


Internal code comments are primarily in Chinese, as this codebase was originally developed for in-lab use. 
An English version of detailed documentation will be added in future updates.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import time
from sklearn.decomposition import PCA

videos = glob.glob(r"C:\Users\Lenovo\Desktop\pycodes\box_0425_200-1000.avi")   #  Video paths are currently hard-coded and should be modified by the user.
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
    std_per = []
    tempmap = []
    shake = False
    bin_size = 2   #越大越稳，但细节抖动会忽略
    shakethreshold1 =1  #越小越精细
    shakethreshold2 = 0.03  #越小越精细
    spatial_decompose = 1 #空间降维参数,越大计算越快

    # 创建VideoWriter来保存结果视频
    output_filename = 'output_result.avi'
    # 使用原始视频的fps，或者根据需要调整
    output_fps = fps / bin_size if fps / bin_size > 0 else 10
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
   
    # 计算缩小后的尺寸
    output_width = int(width/spatial_decompose)
    output_height = int(height/spatial_decompose)
    
    # 创建两个VideoWriter，一个用于std图像，一个用于圆圈检测结果
    out_std = cv2.VideoWriter('std_output.avi', fourcc, output_fps, (output_width, output_height), isColor=False)
    out_circles = cv2.VideoWriter('circles_output.avi', fourcc, output_fps, (output_width, output_height), isColor=True)
    tic = time.time()
    while True:
        ret, curr_frame = capture.read()
        if not ret:
            break
       
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (int(width/spatial_decompose),int(height/spatial_decompose)),interpolation=cv2.INTER_AREA) 
        frames.append(curr_gray)
        frame_count +=1

        if frame_count % bin_size == 0 :
            recent_frame = frames[-bin_size:]   
            recent_diff = cv2.absdiff(recent_frame[0],recent_frame[bin_size-1])              
            recent_frame = recent_frame + [recent_diff]
            recent_stack = np.stack(recent_frame, axis=2)  
            std_recent = np.std(recent_stack,axis=2)
            std_per.append(std_recent)
            
            diff_vs_std = recent_diff.astype(np.float32) - np.mean(std_recent)
            valid_count = np.count_nonzero(diff_vs_std > shakethreshold1)

            if valid_count > shakethreshold2 *width*height:
                shake = True
            else:
                shake = False

        if shake  or frame_count == 1:
            tempmap.append((curr_gray.copy(),frame_count))
            if frame_count != 1:
                print(f"shake detected at frame {frame_count}:{valid_count}pixcles exceed threshold")

        if frame_count % 50 == 0:
            print(f"{frame_count}/{total_frames}")
        
    capture.release()
    # 处理并保存视频
    for i in range(len(std_per)):
        idx = i*bin_size

        mask = tempmap[0][0]

        for temp_frame, temp_idx in tempmap:
            if temp_idx <= idx:
                mask = temp_frame
            else: break
        
        roi = mask * std_per[i]
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
            minRadius=10,   # 最小半径
            maxRadius=30    # 最大半径
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(result_display, (x, y), r, (0, 255, 0), 2)  
                cv2.circle(result_display, (x, y), 2, (0, 0, 255), 3)  
            print(f"{frame_count}: found {len(circles)} circles")
        else:
            print(f"frame {i}:not found")
        
        # 保存帧到视频文件
        out_std.write(roi_display)  # 灰度图像
        out_circles.write(result_display)  # 彩色图像
        
        # 同时显示
        # cv2.imshow('std of videoframes', roi_display)
        # cv2.imshow('circles of videoframes', result_display)
        # cv2.waitKey(50)  
        
    toc = time.time()  
    print(f"执行时间: {toc - tic:.4f} 秒")

    out_std.release()
    out_circles.release()
    cv2.destroyAllWindows()

    print(f"视频已保存为 'std_output.avi' 和 'circles_output.avi'")






# another try: a simple version
    
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import time
from sklearn.decomposition import PCA

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
    std_per = []
    tempmap = []
    shake = False
    bin_size = 2   #越大越稳，但细节抖动会忽略
    shakethreshold1 =1  #越小越精细
    shakethreshold2 = 0.03  #越小越精细
    spatial_decompose = 2 #空间降维参数,越大计算越快

    while True:
        ret, curr_frame = capture.read()
        if not ret:
            break
       
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (int(width/spatial_decompose),int(height/spatial_decompose)),interpolation=cv2.INTER_AREA) 
        frames.append(curr_gray)
        frame_count +=1

        if frame_count % bin_size == 0 :
            recent_frame = frames[-bin_size:]   
            recent_diff = cv2.absdiff(recent_frame[0],recent_frame[bin_size-1])              
            recent_frame = recent_frame + [recent_diff]
            recent_stack = np.stack(recent_frame, axis=2)  
            std_recent = np.std(recent_stack,axis=2)
            std_per.append(std_recent)
            
            diff_vs_std = recent_diff.astype(np.float32) - np.mean(std_recent)
            valid_count = np.count_nonzero(diff_vs_std > shakethreshold1)

            if valid_count > shakethreshold2 *width*height:
                shake = True
            else:
                shake = False
        if shake  or frame_count == 1:
            tempmap.append((curr_gray.copy(),frame_count))
            if frame_count != 1:
                print(f"shake detected at frame {frame_count}:{valid_count}pixcles exceed threshold")

        if frame_count % 50 == 0:
            print(f"{frame_count}/{total_frames}")
        
    capture.release()
    cv2.destroyAllWindows()

    #video_stack = np.stack(frames,axis=2)
    #print(f"a stack of {frame_count} frames has been formed")
    #print(f"shape = {video_stack.shape}")
    #std_image = np.std(video_stack,axis=2)
   

    
    
    for i in range(len(std_per)):
        idx = i*bin_size

        mask = tempmap[0][0]

        for temp_frame, temp_idx in tempmap:
            if temp_idx <= idx:
                mask = temp_frame
            else: break
        
        roi = mask * std_per[i]
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
            minRadius=10,    # 最小半径
            maxRadius=20 # 最大半径
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(result_display, (x, y), r, (0, 255, 0), 2)  
                cv2.circle(result_display, (x, y), 2, (0, 0, 255), 3)  
        else:
            print(f"frame {i}:not found")
        
        cv2.imshow('std of videoframes', roi_display)
        cv2.imshow('circles of videoframes', result_display)
        cv2.waitKey(50)  
"""