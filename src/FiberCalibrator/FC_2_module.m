function outputFolder_fin = FC_2_module(params, outputFolder_1, outputFolder_2)
    %% 本程序主要目的是获得光纤末端闪烁关键帧，作为第二步运行。
    
    % ***重要*** 运行前先检查文件路径、图片路径，并且把FC_1中名称为【_all_.png】 和【im-1-bkg_.png】的两个图片文件删掉。会干扰程序。
    
    % 没有添加交互界面，因此需要手动 运行节【确认模板ROI】，确定参数后再 从头运行。
    
    % 判断运行质量的方法：输出文件 all.png 中是否检测到全部光纤？（只多不少）
    
    outputFolder_fin = params.output_folder_fin;

    imageFiles = dir(fullfile(outputFolder_1, '*.png'));  
    backgrounds = dir(fullfile(outputFolder_2, '*.png'));  
    frameIdx = 1;
    
    %% 确认模板ROI
    
    fiberPlace = imread(fullfile(params.data_folder, params.fc2.fiber_image));      % 读取光纤输出端模板。  （'文件名'，'图片名.格式'）
    pic = fiberPlace;
    
    left = params.fc2.left;                          % 调整left\right\up\down 四个剪裁参数
    right = params.fc2.right;
    up = params.fc2.up;
    down = params.fc2.down;
    
    fiberplace = fiberPlace(up:end-down,left:end-right,:);          % 剪裁
    [centerF, radiiF] = imfindcircles(fiberplace,[5,15], 'Sensitivity', 0.85, 'EdgeThreshold', 0.1);
    centerF(:,1) = centerF(:,1) + left;   
    centerF(:,2) = centerF(:,2) + up;     
    
    pictry = pic(up+1:end-down,left+1:end-right,:);   % 模板ROI区域确认
    figure;
    imshow(pictry);
    
    %% 获取关键帧
    
    for k = 1 :length(imageFiles)-1 
        disp(k);
        fiberPlacek = fiberPlace;
        imagePath = fullfile(outputFolder_1, imageFiles(k).name);
        img = imread(imagePath);
    
        backgroundPath = fullfile(outputFolder_2, backgrounds(k).name);
        bkg = imread(backgroundPath);
    
        [height, width, ~] = size(img);               
       
        lowerHalf = img(floor(height/2)+1 : end ,  : , : ); 
    
        grayImage = rgb2gray(lowerHalf);             % 将下半部分图像转换为灰度图
        brightnessThreshold = params.fc2.brightness_threshold;                    % 亮度阈值：判断是否有闪烁出现。参考范围（15，80），当闪烁较明显时取>50，闪烁暗淡时取<30
       
        brightSpots = grayImage > brightnessThreshold;
        [Y, X] = find(brightSpots);
        [grayHeight, grayWidth] = size(grayImage);
        
        valid = (X > left & X < grayWidth-right) & (Y > up & Y < grayHeight-down);  
        
        intensity = zeros(size(X));
        intensity(valid) = grayImage(sub2ind(size(grayImage), Y(valid), X(valid)));
        disp(['Valid points: ', num2str(sum(valid))]);
    
         % %(可选)debug：是否正确检测亮点
         % normalizedIntensity = double(intensity) / 255;    %归一化亮度值
         % 
         % if ~isempty(X)
         %    for i = 1:length(X)
         %        posi = [X(i), Y(i)+(height/2), 5];    % 每个亮斑的位置和半径
         %        opacity = normalizedIntensity(i);     % 当前亮斑的透明度
         %        img = insertShape(img, 'FilledCircle', posi, 'Color', 'yellow', 'Opacity', opacity);
         %    end
         % end
         % figure;
         % imshow(img);
    
         radius = 5;                                                            % 设置搜索半径（闪烁半径），一般无需更改
         numPoints = length(X);                                                 % 计算半径内明亮像素
         weightedNeighborCount = zeros(numPoints, 1);                           % 初始化加权半径内像素数量
         validIndices = find(valid);                                            % 找到 valid 范围内的亮点的索引
    
         for i = 1:numPoints
             % 只对 valid 范围内的点进行计算
             if valid(i)
                 distances = sqrt((X(validIndices) - X(i)).^2 + (Y(validIndices) - Y(i)).^2);
                 inRadius = distances <= radius;            
                 weightedNeighborCount(i) = sum(intensity(validIndices(inRadius)));       % 累计半径内的点的亮度（使用亮度作为权重）
             else          
                 weightedNeighborCount(i) = 0;                                            % 对于 valid 范围外的点，加权邻居数量为 0
             end
         end
           
         [~, maxIdx] = max(weightedNeighborCount);                                        % 找到闪烁半径内明亮像素密度最高的点
     
         if max(weightedNeighborCount) ~= 0
             mostDenseX = X(maxIdx);   % 最密处点坐标
             mostDenseY = Y(maxIdx);
         else
             mostDenseX = 0; % 若不存在则赋0值
             mostDenseY = 0;
         end  
            
    
        if mostDenseX ~=0 && mostDenseY ~=0 
            pos = [mostDenseX,mostDenseY+(height/2),radius]; % 构造闪点数据
            shiner = [mostDenseX,mostDenseY,radius];         
        
            img = insertShape(img, 'FilledCircle', pos, 'Color', 'yellow', 'Opacity', 1);                   %  在视频帧中用黄色标出闪点位置
            fiberPlacek = insertShape(fiberPlacek, 'FilledCircle', shiner, 'Color', 'red', 'Opacity', 1);   % 在模板图中用红色标出闪点位置
            combinedImage = [img; bkg; fiberPlacek];
            %combinedImage = cat(1, img, fiberPlacek);%（可选）合成一张图，不推荐
    
            outputFileName = fullfile(outputFolder_fin, sprintf('match_%04d.png', frameIdx)); % 文件名
            imwrite(combinedImage, outputFileName);  
            pic = insertShape(pic, 'FilledCircle', shiner, 'Color', 'red', 'Opacity', 1); % 在模板图中用红色标出闪点位置（累计）
    
            frameIdx = frameIdx + 1;        % 更新帧索引
        end   
    end
     
    figure;
    imshow(pic);
    
    
    disp(['检测完成，共保存重合帧数量：', num2str(frameIdx - 20)]);
    
    fprintf('FC_2 处理完成\n');
end