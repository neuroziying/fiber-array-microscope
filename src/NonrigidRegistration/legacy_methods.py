import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
import time

videos = glob.glob('C:/Users/Administrator/Desktop/pycodes/box_0425_200-1000.avi')
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
    std_per5 = []
    spatial_decompose = 2 #空间降维参数
    """ 
    ret, prev_frame = capture.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(prev_gray)
    frame_count +=1
    """

    while True:
        ret, curr_frame = capture.read()
        if not ret:
            break
       
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (int(width/spatial_decompose),int(height/spatial_decompose)),interpolation=cv2.INTER_AREA) 
        #curr_gray = cv2.GaussianBlur(curr_gray, (15, 15), 0)
        frame_count +=1


        # Method 1 (default): Frame-wise Hough circle detection.
        # pros: easy, clean, fast
        # suitable for clear fiber cross-sections

        ccls = cv2.HoughCircles(
            curr_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,           # 累加器分辨率与图像分辨率的反比
            minDist=30,     # 检测到的圆心之间的最小距离
            param1=50,      # Canny边缘检测的高阈值
            param2=30,      # 圆心检测阈值
            minRadius=5,    # 最小半径
            maxRadius=20  # 最大半径
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



    # ===== Primary method: STD-based detection =====
    # STD-based fiber localization pipeline (primary method).
    # Pipeline: video -- grayscale -- temporal STD -- ROI normalization -- Hough circle detection
    # Suitable for noisy or low-resolution data
    frames = []
    frame_count = 0
    for i in range(len(std_per5)):
        roi = std_image * std_per5[i]

        roi_normalized = (roi - roi.min()) / (roi.max() - roi.min()) * 255
        roi_display = roi_normalized.astype(np.uint8)
        result_display = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(
            roi_display,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=30
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(result_display, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result_display, (x, y), 2, (0, 0, 255), 3)
        else:
            print(f"{frame_count}: not found")

        cv2.imshow('std of videoframes', roi_display)
        cv2.imshow('circles of videoframes', result_display)
        cv2.waitKey(50) 


    # ===enhanced version =======
    # STD-based pipeline with temporal weighting (enhanced version).
    # Suitable for noisy or low-resolution data
    # cons: time consuming
    
    frames = []

    while True:
        ret, curr_frame = capture.read()
        if not ret:
            break
        """
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        std_diff = np.std(frame_diff)
        if len(frame_diff > std_diff)> :
             cv2.imshow('diff',frame_diff)
             cv2.waitKey(50)
        prev_gray = curr_gray
        """
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(
            curr_gray,
            (int(width / spatial_decompose), int(height / spatial_decompose)),
            interpolation=cv2.INTER_AREA
        )
        frames.append(curr_gray)

    video_stack = np.stack(frames, axis=2)
    std_image = np.std(video_stack, axis=2)
    
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
    for i in range(len(std_per5)):
        roi = std_image * std_per5[i]

        roi_normalized = (roi - roi.min()) / (roi.max() - roi.min()) * 255
        roi_display = roi_normalized.astype(np.uint8)
        result_display = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(
            roi_display,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=30
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(result_display, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result_display, (x, y), 2, (0, 0, 255), 3)
        else:
            print("not found")

        cv2.imshow('std of videoframes', roi_display)
        cv2.imshow('circles of videoframes', result_display)
        cv2.waitKey(50)
