function [outputFolder_1, outputFolder_2] = FC_1_module(params)
    
    % 使用params中的参数
    outputFolder_1 = params.output_folder_1;
    outputFolder_2 = params.output_folder_2;   
    
    CirclePlace = imread(fullfile(params.data_folder, params.fc1.circle_image));  % 读取光纤图像（进光端/圆圈半径较大的一端）
    I_eq = CirclePlace;
    
    %% 增强对比度（可选，背景暗淡时启用，一般不用）
    
    % I_grey = rgb2gray(CirclePlace);
    % [counts,x] = imhist(I_grey);
    % p = counts / numel(I_grey);
    % cdf = cumsum(p);
    % map = uint8(255*cdf);
    % I_eq = map(I_grey+1);
    % subplot(1,2,1),imshow(CirclePlace);
    % subplot(1,2,2),imshow(I_eq);
    
    %% 确定ROI
    
    left = params.fc1.left;                                      % left\right\up\down四个参数需要调节。
    right = params.fc1.right;                                     %目标是框选出关注的区域。四个参数分别代表上下左右减裁掉的距离
    up = params.fc1.up;
    down = params.fc1.down;
    
    cptry = I_eq( up+1 :end-down , left+1 :end-right , : );   %更改区间，框选出ROI    （y1:y2,x1:x2,rgb）
    imshow(cptry);                                      %展示ROI，确定区间
    
    
    %% 得到光纤背景
    
    cp = I_eq(up+1 :end - down,left+1 :end-right,:);              %将确定的ROI区间输入
    [centerA, radiiA] = imfindcircles(cp,[25,30], 'Sensitivity', 0.98,'EdgeThreshold',0.1,'ObjectPolarity','bright'); % [25,30]:半径范围
                                                                                        % sensitivity：圆形图案敏感度，一般不取很高（0.5~0.99)，识别效果不佳则调高此值。
                                                                                        % edgethreshold： 圆形边界敏感度，一般不取很低，（0.05~0.3），如果识别到噪声圆圈则调高此值
                                                                                        % 当目标圆圈比背景亮时OP取'bright'，当目标圆圈比背景暗时OP取'dark'
    if ~isempty(centerA)
        centerA(:,1) = centerA(:,1)+left;              %坐标变换为ROI系
        centerA(:,2) = centerA(:,2)+up;
    end
    Circles = [centerA,radiiA];
    disp(length(Circles));
    circles = insertShape(CirclePlace, 'Circle', Circles, 'Color', 'blue', 'LineWidth', 3);
    imshow(circles);
    %% 基础参数设置
    
    k = 0;
    videoObj = VideoReader(fullfile(params.data_folder, params.fc1.video_file));     % 读取目标视频。   '文件夹名称'+'视频名称.格式'
    frameRate = videoObj.FrameRate;
    startTime = 5 / frameRate;                                 % 设置开始读取的时间（处理首帧缺失情况，一般无需更改）
    videoObj.CurrentTime = startTime;
    bkf = readFrame(videoObj);
    
    pic = read(videoObj,1);
    frameIdx = 25;                                              % 设置开始帧，与startTime关联，一般无需更改
    threshold = params.fc1.threshold;                                                        % 重叠判定：光圈与光纤圆心间最短距离（pix）。一般无需更改
    brightness_threshold_L = params.fc1.brightness_threshold_L;                              % 最低亮度阈值：判定是否存在光照。一般无需更改。（20~60）
    brightness_threshold_H = params.fc1.brightness_threshold_H;                              % 最高亮度阈值：光纤被照射后亮起的最大亮度。一般无需更改 （230~255）
    
    % 上述2个亮度阈值取值在合理范围内即可；区间极端取值影响计算速度。
    
    %% 视频ROI参数确认
    
    [height, width,~] = size(bkf);
    left_indent = params.fc1.LI;                                        % 框选视频ROI x区间（左端）
    right_indent = params.fc1.RI;                                       % 框选视频ROI x区间（右端）
    upbkf = bkf(1:floor(height/2), left_indent:width-right_indent, :);
    
    figure;                                                             % 展示ROI,确认参数
    imshow(upbkf);
    gray1 = im2gray(upbkf);
    imshow(gray1);
    
    %% 提取"光斑照入光纤"的关键帧
    
    while hasFrame(videoObj)
        k = k+1;
        disp(k);                            
        frame = readFrame(videoObj);
        background = circles;
    
        [height, width,~] = size(frame);
        upperHalf = frame(1:floor(height/2), left_indent:width-right_indent, :);
        gray2 = im2gray(upperHalf);
        brightness_increase = gray2 - gray1;                         % 得到差值矩阵
    
        mask = (brightness_increase > brightness_threshold_L) & ...
            (brightness_increase < brightness_threshold_H);
    
        se = strel('disk', 10);                                      % 结构元素大小根据噪声情况调整，一般无需改动
        mask = imopen(mask, se);                                     % 去除噪点
    
        if any(mask(:))
            % figure;                                                %（可选）确认光斑位置
            % imshow(brightness_increase);
            [centerB, radiiB] = imfindcircles(mask, [10, 25], 'Sensitivity', 0.97);
    
    
            % 如果检测到光斑
            if ~isempty(centerB)
                meanCenter = mean(centerB, 1);                           % 计算圆心均值
                circleData = [meanCenter, max(radiiB)];                  % 使用圆心和半径的均值
    
                % figure;                                                %（可选）确认光斑被正确检测
                % bricir = insertShape(brightness_increase, 'Circle', circleData, 'Color', 'red', 'LineWidth', 3);
                % imshow(bricir);
                
    
                %判断是否重合
                distance = sqrt((circleData(1)+ left_indent - centerA(:, 1)).^2 + (circleData(2) - centerA(:, 2)).^2);
                [minDist, idx] = min(distance);  % 找到最小距离和其索引
    
    
                closestCenter = centerA(idx, :);  % 获取最小距离对应的圆心
                closestCircle = [closestCenter,radiiA(idx)];
    
    
                if minDist <= threshold
                    pic = insertShape(pic, 'Circle', closestCircle, 'Color', 'green', 'LineWidth', 3);
                    frame_with_circles = insertShape(frame, 'Circle', [circleData(:,1)+left_indent, circleData(:,2), circleData(:,3)], 'Color', 'red', 'LineWidth', 3);
                    frame_with_circles_RY = insertShape(frame_with_circles, 'Circle', closestCircle, 'Color', 'yellow', 'LineWidth', 3);
                    outputFileName_1 = fullfile(outputFolder_1, sprintf('_4_%04d.png', frameIdx)); % 保存帧到输出文件夹
                    imwrite(frame_with_circles_RY, outputFileName_1);
    
                    background = insertShape(background, 'Circle', closestCircle, 'Color', 'yellow', 'LineWidth', 3);
                    outputFileName_2 = fullfile(outputFolder_2, sprintf('_4_%04d.png', frameIdx)); % 保存帧到输出文件夹
                    imwrite(background, outputFileName_2);
                    frameIdx = frameIdx + 1;% 更新帧索引
                end
            end
        end
    end
    
    figure;
    imshow(pic);
    
    
    disp(['检测完成，共保存重合帧数量：', num2str(frameIdx - 20)]);
    
    fprintf('FC_1 处理完成\n');
end