
outputFolder= 'test-box';       % 保存路径
if ~exist('test-box', 'dir')
    mkdir('test-box');
end
%读取视频
start_frame = 1;  %断点续跑
videoObj = VideoReader(fullfile('box_0425_200-1000.avi'));
frameRate = videoObj.FrameRate; % 帧率
videoObj.CurrentTime = (start_frame - 1) / videoObj.FrameRate;
frame0 = readFrame(videoObj);
first_pic = frame0(1:end,:,:);  %根据视频比例调整
[h,w,~] = size(first_pic);   %背景基准

%读取数据
dataP = readtable(fullfile('100-12/','output-12.xlsx'));
X = table2array(dataP(:, 2:3)); 
X = [w - X(:,1), X(:,2)];  %得到模板坐标

j = start_frame - 1;  % 继续计数
shinningthreshold = 100; 
radius = 15;
brightness_matrix = [];
totalTime = 0; 
hundred_frame_costTime = 0;
params0 = [3.204, 840, 780, 0.76];  % 初始猜测（由tmptry得到）
frame_block = 100;
num_points = size(X,1);
brightness_block = zeros(frame_block, num_points);
brightness_row = zeros(1, num_points); 
% figureHandle = figure('Name', '配准结果');
% ax = axes('Parent', figureHandle);  % 创建 axes 子区域
% frame_handle = imshow(zeros(size(frame0), 'like', first_pic), 'Parent', ax);

checker = 0;
outputVideo = VideoWriter('box.avi', 'Motion JPEG AVI');  % 兼容 Fiji
open(outputVideo);


%%主函数

while hasFrame(videoObj)
    j=j+1;
    block_index = mod(j-1, frame_block) + 1; 
    frame = readFrame(videoObj);
    imgs = frame(1:end,:,:);  %根据视频比例调整
    tic;
    if j==start_frame
        last_pic = imgs;
        last_shake = movement_detector(last_pic); 
        continue;
    else 
        this_shake = movement_detector(imgs); %排除剧烈抖动帧
        shift_left  = abs(this_shake(1) - last_shake(1));
        shift_right = abs(this_shake(2) - last_shake(2));
        down_shake = abs(this_shake(3)-last_shake(3));
        up_shake = abs(this_shake(4)-last_shake(4));
        last_shake = this_shake;
        if shift_left > 100 || shift_right > 100 || down_shake > 100 || up_shake > 100
            disp('跳过异常抖动帧');
            continue;

        else
            if shift_left < 2 && shift_right < 2 && down_shake < 2 && up_shake < 2
                disp('原帧小抖动');
                if checker == 0
                    checker = checker+1;
                    plained_imgs = frame_diff_mask(imgs, last_pic, shinningthreshold); % 频闪修正
                    img = plained_imgs;
                    [Y,radii] = edge_detector(img);  % 当前帧边缘识别
                    try
                        opt_params = fminsearch(@(p) transform_cost(p, X, Y), params0, optimset('MaxFunEvals', 5000, 'Display', 'off'));
                    catch ME
                        warning('第 %d 帧配准失败：%s，跳过该帧。', j, ME.message);
                        continue;  % 跳过当前帧
                    end
                    theta = opt_params(1);
                    tx = opt_params(2);
                    ty = opt_params(3);
                    scale = opt_params(4);
                    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
                    T = [tx; ty];

                    X_trans = (scale * R * X') + T;
                    X_trans = X_trans'; % 配准后模板坐标

                    brightness_row = get_value(X_trans, img, radius);

                    % 存储
                    img_circle = insertShape(frame, 'circle', [X_trans(:,1), X_trans(:,2),  repmat(radius, size(X_trans,1), 1)], 'Color', 'green', 'Opacity', 1,'LineWidth',5);
                    writeVideo(outputVideo, img_circle);
                    params0 = opt_params;
                    
                else
                    plained_imgs = frame_diff_mask(imgs, last_pic, shinningthreshold); % 频闪修正
                    img = plained_imgs;
                    brightness_row = get_value(X_trans, img, radius);
                    img_circle = insertShape(frame, 'circle', [X_trans(:,1), X_trans(:,2),  repmat(radius, size(X_trans,1), 1)], 'Color', 'green', 'Opacity', 1,'LineWidth',5);
                    writeVideo(outputVideo, img_circle);
                    
                end
            else
                disp('原帧大抖动');
                checker = checker+1;
                plained_imgs = frame_diff_mask(imgs, last_pic, shinningthreshold); % 频闪修正
                img = plained_imgs;

                [Y,radii] = edge_detector(img);  % 当前帧边缘识别


                try
                    opt_params = fminsearch(@(p) transform_cost(p, X, Y), params0, optimset('MaxFunEvals', 5000, 'Display', 'off'));
                catch ME
                    warning('第 %d 帧配准失败：%s，跳过该帧。', j, ME.message);
                    continue;  % 跳过当前帧
                end
                theta = opt_params(1);
                tx = opt_params(2);
                ty = opt_params(3);
                scale = opt_params(4);
                R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
                T = [tx; ty];

                X_trans = (scale * R * X') + T;
                X_trans = X_trans'; % 配准后模板坐标

                brightness_row = get_value(X_trans, img, radius);
                %brightness_matrix(j, :) = brightness_row;

                % 存储
                img_circle = insertShape(frame, 'circle', [X_trans(:,1), X_trans(:,2),  repmat(radius, size(X_trans,1), 1)], 'Color', 'green', 'Opacity', 1,'LineWidth', 5);
                writeVideo(outputVideo, img_circle);
                

                params0 = opt_params;
                fprintf('%d帧配准完成\n', j);


             end
             last_pic = imgs;
             
             dt = toc;
             totalTime = totalTime + dt;
             hundred_frame_costTime = hundred_frame_costTime+dt;
             fprintf('帧耗时：%.3f 秒\n', dt);
        end

        brightness_block(block_index, :) = brightness_row;
        if block_index == frame_block || ~hasFrame(videoObj)
            fprintf('平均耗时：%.3f 秒\n', hundred_frame_costTime / block_index);
            filename = sprintf('brightness_%04d.mat', j);
            save(fullfile(outputFolder, filename), 'brightness_block', '-v7.3');
            fprintf('保存了第 %d 帧\n', j);

            % 重置
            brightness_block = zeros(frame_block, num_points);
            block_index = 0;
            hundred_frame_costTime = 0;
            % 清理变量、释放内存
            drawnow;
            pause(1);
        end
    end
end
fprintf('处理完成，平均每帧耗时：%.3f 秒\n', totalTime / j);
% writematrix(brightness_matrix, 'output-test.xlsx', 'Sheet', 1, 'Range', 'A1');
all_data = [];
files = dir(fullfile(outputFolder, 'brightness_*.mat'));
for k = 1:length(files)
    data = load(fullfile(files(k).folder, files(k).name));
    all_data = [all_data; data.brightness_block];
end
save('box.mat', 'all_data');
writematrix(all_data, 'box.xlsx'); 
close(outputVideo);
%%
function plained_imgs = frame_diff_mask(imgs, lastpic, threshold)  % 频闪修正
    gray_diff = abs(double(rgb2gray(imgs)) - double(rgb2gray(lastpic)));
    mask = gray_diff > threshold;
    plained_imgs = imgs;
    plained_imgs(repmat(mask, [1, 1, 3])) = 0;
end



function shake = movement_detector(imgs)  %拍摄波动修正
    t1 = 86.50;
    t2 = 55.90;    %由tmptry得到
    t3 = 38.05;
    x1 = 266;
    x2 = 1024;
    valid = imgs(1:end,x1:x2,:);
    red_mask = (valid(:,:,1)>t1 & valid(:,:,2)<t2 & valid(:,:,3)<t3 );
    [rows, cols] = find(red_mask);  % 找出所有真值像素位置
    x_min = min(cols);  
    x_max = max(cols); 
    y_min = min(rows);
    y_max = max(rows);
    shake = [x_min,x_max,y_min,y_max];  %水平抖动
end



function [center,radii] = edge_detector(img)   %边缘检测函数
thresh = 0.13;  %由tmptry确定
sensitivity = 0.98; %由tmptry确定
edge_img = edge(rgb2gray(img), 'canny', [thresh, 2*thresh], 1.5);
[center, radii] = imfindcircles(edge_img,[25,30], 'Sensitivity',sensitivity, 'EdgeThreshold', 0.1);
end


function cost = transform_cost(params, X, Y)    %配准主函数
    theta = params(1);       % 旋转角度
    tx = params(2);          % x方向平移
    ty = params(3);          % y方向平移
    s = params(4);           % 缩放因子

    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    T = [tx; ty];

    X_trans = (s * R * X') + T;   % X 转置后是 2×N，乘完再加 T
    X_trans = X_trans';           % 再转置回来 N×2

    [~, dists] = knnsearch(Y, X_trans);     % 最近邻匹配（每个 X_trans 对应 Y 中最近的点）

    % 离群点剔除
    dists = sort(dists);  
    M = round(0.9 * length(dists));   % 取前90%的小误差点
    cost = mean(dists(1:M).^2);       % cost函数定义
end


function brightness_row = get_value(X_trans, imgs, radius )

    gray_img = rgb2gray(imgs);
    [H, W] = size(gray_img);

    N = size(X_trans, 1);
    brightness_row = zeros(1,N);
    [X_grid, Y_grid] = meshgrid(1:W, 1:H);
    
    %过滤红色边缘
    t1 = 86.50;
    t2 = 55.90;    %由tmptry得到
    t3 = 38.05;
    red_cutter = (imgs(:,:,1) > t1) & (imgs(:,:,2) < t2) & (imgs(:,:,3) < t3);

    for i = 1:N
        cx = X_trans(i, 1);
        cy = X_trans(i, 2);
        r = radius;  
        circle_mask = (X_grid - cx).^2 + (Y_grid - cy).^2 <= r^2;
        valid_mask = circle_mask & ~red_cutter;

        values = gray_img(valid_mask);

        if ~isempty(values)
            brightness_row(i) = mean(values);
        else
            brightness_row(i) = 0;  
        end
    end
end
