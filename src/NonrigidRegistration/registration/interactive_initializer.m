  outputFolder = 'test_detection';
    videoPath = fullfile('im-1','HDFP0.AVI');    
    % 执行视频分析
    video_edge_analysis(videoPath, outputFolder);
    

%% 视频分析主函数
function video_edge_analysis(videoPath, outputFolder)
    % 初始化
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    videoObj = VideoReader(videoPath);
    first_pic = readFrame(videoObj);
    [h,w,~] = size(first_pic);
    img = first_pic(1:end,:,:);
    dataP = readtable(fullfile('100-12/','output-12.xlsx'));
    X = table2array(dataP(:,2:3)); 
    X = [w - X(:,1), X(:,2)];

    %创建交互界面
    fig = figure('Name', '视频分析', 'Position', [10 10 2400 600]);
    ax1 = subplot(1,3,1); title('剪裁后图像');
    ax2 = subplot(1,3,2); title('红色区域检测');
    ax3 = subplot(1,3,3); title('边缘检测');

    %添加控制滑块
    uicontrol('Style', 'slider', 'Min',0, 'Max',255, 'Value',10,  'Position',[10 20 100 20], 'Tag','red');
    uicontrol('Style', 'slider', 'Min',0, 'Max',255, 'Value',10,  'Position',[10 50 100 20], 'Tag','green');
    uicontrol('Style', 'slider', 'Min',0, 'Max',255, 'Value',10,  'Position',[10 80 100 20], 'Tag','blue');
    uicontrol('Style', 'slider', 'Min',0, 'Max',1,   'Value',0.1, 'Position',[10 140 100 20], 'Tag','try2');
    uicontrol('Style', 'slider', 'Min',0, 'Max',w,   'Value',10,  'Position',[120 20 100 20], 'Tag','x1');
    uicontrol('Style', 'slider', 'Min',0, 'Max',w,   'Value',w,   'Position',[120 50 100 20], 'Tag','x2');
    uicontrol('Style', 'slider', 'Min',0, 'Max',1, 'Value',0.01,'Position',[120 80 100 20], 'Tag','sensitivity');
    uicontrol('Style', 'slider', 'Min',0, 'Max',2*pi, 'Value',0,  'Position',[240 20 100 20], 'Tag','theta');
    uicontrol('Style', 'slider', 'Min',-1000, 'Max',1000, 'Value',0,  'Position',[240 50 100 20], 'Tag','tx');
    uicontrol('Style', 'slider', 'Min',-1000, 'Max',1000, 'Value',0,  'Position',[240 80 100 20], 'Tag','ty');
    uicontrol('Style', 'slider', 'Min',0, 'Max',2, 'Value',0.5,  'Position',[240 110 100 20], 'Tag','scale');
    while isvalid(fig)  % 若窗口被关闭，则自动退出
        sensitivity = get(findobj(fig, 'Tag','sensitivity'), 'Value');
        x1 = round(get(findobj(fig, 'Tag','x1'), 'Value'));
        x2 = round(get(findobj(fig, 'Tag','x2'), 'Value'));
        t1 = get(findobj(fig, 'Tag','red'), 'Value');
        t2 = get(findobj(fig, 'Tag','green'), 'Value');
        t3 = get(findobj(fig, 'Tag','blue'), 'Value');
        thresh = get(findobj(fig, 'Tag','try2'), 'Value');

   
        % 处理流程1: 有效区检测       
        valid = img(:,x1:x2,:);
       
        % 处理流程2: 红色区域检测       
        red_mask = (valid(:,:,1)>t1 & valid(:,:,2)<t2 & valid(:,:,3)<t3 );
        [centerR, radiiR] = imfindcircles(red_mask,[25,30], 'Sensitivity',sensitivity, 'EdgeThreshold', 0.1);
        if ~isempty(centerR)
            imgR = insertShape(repmat(red_mask, [1,1,3])*255, 'circle', [centerR(:,1), centerR(:,2), radiiR], 'Color', 'green', 'Opacity', 1);
        else
            imgR = repmat(red_mask, [1,1,3])*255;
        end
    
        % 处理流程3: 边缘检测
        edgeR = edge(rgb2gray(valid), 'canny', [thresh, 2*thresh], 1.5);
        [center, radii] = imfindcircles(edgeR,[25,30], 'Sensitivity',sensitivity, 'EdgeThreshold', 0.1);
        if ~isempty(center)
            imgE = insertShape(repmat(edgeR, [1,1,3])*255, 'circle', [center(:,1), center(:,2), radii], 'Color', 'green', 'Opacity', 1);
        else
            imgE = repmat(edgeR, [1,1,3])*255;
        end
       %流程4：预配准
       theta = get(findobj(fig, 'Tag','theta'), 'Value');
       tx = get(findobj(fig, 'Tag','tx'), 'Value');
       ty = get(findobj(fig, 'Tag','ty'), 'Value');
       scale = get(findobj(fig, 'Tag','scale'), 'Value');
       R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
       T = [tx; ty];
       X_trans = (scale * R * X') + T;
       X_trans = X_trans';
       img_circle = insertShape(img, 'circle', [X_trans(:,1), X_trans(:,2),  repmat(25, size(X_trans,1), 1)], 'Color', 'green', 'Opacity', 1);


       % 显示结果
       imshow(img_circle, 'Parent', ax1);
        title(ax1, sprintf('配准展示(theta=%d,tx=%d,ty=%d,scale=%.2f)',theta,tx,ty,scale));
        imshow(imgR , 'Parent', ax2);
        title(ax2, sprintf('红色区域(r=%.2f,r=%.2f,r=%.2f,x1=%d,x2=%d)',t1,t2,t3,x1,x2));
        imshow(imgE, 'Parent', ax3);
        title(ax3, sprintf('边缘检测(阈值=%.2f,sen=%.2f)', thresh,sensitivity));
        drawnow;
     end
end


% 
% function cleaned_imgs = cleaning(imgs, threshold)
% grayImage = rgb2gray(imgs);
% brightSpots = grayImage > threshold;
% [Y, X] = find(brightSpots);
% radius = 5;% 设置搜索半径
% for i = 1:length(brightSpots)
%     distances = sqrt((X(validIndices) - X(i)).^2 + (Y(validIndices) - Y(i)).^2);
%     inRadius = distances <= radius;
%     weightedNeighborCount(i) = sum(intensity(validIndices(inRadius))); % 累计半径内的点的亮度（使用亮度作为权重）
% 
%     weightedNeighborCount(i) = 0;   % 对于 valid 范围外的点，加权邻居数量为 0
% end
% mask = diff > threshold;
% tmp = imgs(:,:,k);
% tmp(mask) = 0;
% cleaned_imgs(:,:,k) = tmp;
% end