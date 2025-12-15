function FC_4_module(params, outputFolder_fin, invalidIndices)
    % FC_4 的模块化版本 - 输出最终结果
    
        %% 本程序输出最终结果。不论FC_3效果如何，都要运行FC_4以获得excel输出
    
    
    outputFolder = params.output_folder_final;       % 保存结果的文件夹
    inputFolder = outputFolder_fin;             % 设置读取路径，应与FC_2的输出文件夹一致
    if ~ischar(outputFolder) && ~isstring(outputFolder)
        outputFolder = char(outputFolder);
    end
    if ~ischar(inputFolder) && ~isstring(inputFolder)
        inputFolder = char(inputFolder);
    end
    imageFiles = dir(fullfile(inputFolder, '*.png'));  
    
    %% 确定ROI
    circlePlace = imread(fullfile(params.data_folder,  params.fc3.circle_image));  % 读取光纤输入端模板（圆圈半径更大的那一张）
    circles = circlePlace;
    
    fiberPlace = imread(fullfile(params.data_folder, params.fc3.fiber_image));  % 读取光纤输出端模板（圆圈半径更小的那一张）
    
    left = params.fc3.left;                          
    right = params.fc3.right;                         % ***重要*** 这里的参数应和 FC_2 中取相同的值
    up = params.fc3.up;
    down = params.fc3.down;
    
    fiberPlace = fiberPlace(up+1:end-down,left+1:end-right,:);          % 剪裁
    
    [centerF, radiiF] = imfindcircles(fiberPlace,[5,30], 'Sensitivity', 0.7, 'EdgeThreshold', 0.1);
    centerF(:,1) = centerF(:,1) + left;  
    centerF(:,2) = centerF(:,2) + up;  
    r = sqrt(centerF(:,1).^2+centerF(:,2).^2);
    [sortedr,idx] = sort(r);
    sortedF = centerF(idx,:);            % sortedF里保存了排序好的光纤末端坐标
    q = fiberPlace;
    fiber = fiberPlace;
    for i = 1:length(sortedF)
        fiber = insertText(fiber, [sortedF(i,1)-left,sortedF(i, 2)], num2str(i), 'TextColor', 'red', 'FontSize', 20, 'BoxOpacity', 0);
    end
    output3 = fullfile(outputFolder, sprintf('sorted_fibers.png'));   % 光纤末端排序，保存。
    imwrite(fiber, output3);
    
    minDist = inf;

    
    %% 开始定标
    
    numImages = length(imageFiles);
    redPointData = struct('Center_point', {}, 'Center_yellow', {}, 'Image', {}, 'length', {});       
    moreredPointData = struct('Center_point', {}, 'Center_yellow', { }, 'Image', {}, 'length', {});   
    w = 0;
    z = 0;
    
    % === 第一步：预处理所有图像，提取红点信息 ===
    disp('Preprocessing images...');
    for k = 1:numImages
        disp(k);
        imagePath = fullfile(inputFolder, imageFiles(k).name);
        img = imread(imagePath);
        [height, width, ~] = size(img);
        
        % 划分区域
        lowerQuarter = floor(height * 3 / 4):height;  
        higherQuarter = 1:floor(height * 1 / 4);
        
        if size(img, 3) == 3
            % 提取 RGB 通道
            R = img(:, :, 1);
            G = img(:, :, 2);
            B = img(:, :, 3);
    
            % 检测红色点
            R_lower = R(lowerQuarter, :);
            G_lower = G(lowerQuarter, :);
            B_lower = B(lowerQuarter, :);
            redPoints = (R_lower > 230) & (G_lower < 100) & (B_lower < 100);     %（可选）'红色'判定范围，一般无需更改
            [Y, X] = find(redPoints);
            
            if ~isempty(Y) && ~isempty(X) && length(X) > 5
                % 红点的中心位置
                centerX = mean(X);
                centerY = mean(Y);
            end
    
            % 检测黄色圈
            R1 = R(higherQuarter, :);
            G1 = G(higherQuarter, :);
            B1 = B(higherQuarter, :);
            yellowMask = (R1 == 255) & (G1 == 255) & (B1 == 0);                   %（可选）'黄色'判定范围，一般无需更改
            [centersY, radiiY] = imfindcircles(yellowMask, [10, 50], 'Sensitivity', 0.9);      %（可选）检测黄圈标记，一般无需更改
            
            if ~isempty(centersY)
                radiiyY = mean(radiiY);
                centeryX = centersY(1, 1);  
                centeryY = centersY(1, 2)-5;  
            end
            
            
            % 检测红色圈
            redMask = (R1 == 255) & (G1 == 0) & (B1 == 0);    %（可选）'红色'判定范围，一般无需更改
            [centersR, radiiR] = imfindcircles(redMask, [10, 50], 'Sensitivity', 0.9);   %（可选）检测红圈标记，一般无需更改
            if ~isempty(centersY) && ~isempty(centersR)
                dyr = sqrt((centersY(1)-centersR(1)).^2+(centersY(2)-centersR(2)).^2);
                moreredPointData(k) = struct('Center_point', [centerX, centerY,5],'Center_yellow', [centeryX,centeryY,radiiyY],'Image', img,'length',dyr);
                z = z+1; 
            end
    
    
    
            % 创建外圈掩膜
            [xx, yy] = meshgrid(1:width, 1:height);
            if ~isempty(centersR)
            circleMask = ((xx -  centeryX).^2 + (yy - centeryY).^2) <= radiiyY^2;
            outsideCircleMask = ~circleMask;
            else
                w = w+1;
               dYR = 1 ;
               redPointData(w) = struct('Center_point', [centerX, centerY,5],'Center_yellow', [centeryX,centeryY,radiiyY],'Image', img,'length',dYR);
               disp('notfound_redcircle');
                continue
            end
            % 提取 higherQuarter 区域图像
            higherQuarterImage = img(higherQuarter, left :end-right, :);
            
            % 转换为灰度图
            if size(higherQuarterImage, 3) == 3
                higherQuarterGray = rgb2gray(higherQuarterImage);
            else
                higherQuarterGray = higherQuarterImage;
            end
            
            % 裁剪外圈掩膜到 higherQuarter
            higherQuarterMask = outsideCircleMask(higherQuarter, left:end-right);  
            
            % 应用掩膜并提取亮点
            maskedBrightness = higherQuarterGray .* uint8(higherQuarterMask);
            [ly, lx] = find(maskedBrightness > params.fc3.brightness_threshold);    
            
            % 显示亮点信息
            disp(['找到的亮点数量: ', num2str(length(lx))]);
    
            if length(lx) <params.fc3.spot_count_threshold          % 【取和FC_3中一样的值即可】        
                                                         
               w = w+1;
               dYR = sqrt((centersY(1)-centersR(1)).^2+(centersY(2)-centersR(2)).^2);
               redPointData(w) = struct('Center_point', [centerX, centerY,5],'Center_yellow', [centeryX,centeryY,radiiyY],'Image', img,'length',dYR);
               
            else
                disp('亮点数量超过阈值，忽略。');
            end
        end
    end
    fprintf('处理完成！w = %d\n', w);
    
    %% === 第二步：匹配光纤末端和红点 ===
    
    more = zeros(1, length(sortedF));    % 预分配数组
    count = 0;                           % 记录不满足条件的计数器
    disp('Matching fiber endpoints with red points...');
    allData = [];
    for i = 1:length(sortedF)
        data = [];
        minD = inf;
        bestMatch = []; % 存储最佳匹配的图像
        best_red_point = [];
        bestcenters = [];
        matches = struct('d', {}, 'dYR', {}, 'img', {}, 'redCenter', {}, 'centers', {});
    
        if ismember(i, invalidIndices)
            continue
        end
        
        for k = 1:w
            % 获取红点数据
            redCenter = redPointData(k).Center_point;
            img = redPointData(k).Image;
            dYR = redPointData(k).length;
            centers = redPointData(k).Center_yellow;
            
            % 计算光纤末端和红点中心的距离
            d = sqrt((sortedF(i, 1) - redCenter(1))^2 + (sortedF(i, 2) - redCenter(2))^2);
    
            %更新最小距离和最佳匹配
            if d < minD
                minD = d;
                bestMatch = img;
                best_red_point = redCenter;
                bestcenters = centers;
            end      
        end
    
        if ~isempty(bestMatch)
            [height, ~] = size(bestMatch);       
            bestfiber = bestMatch(floor(height*3/4)+1 : end, : , :);
            bestpic = bestMatch(floor(height*1/4)+1 : floor(height*1/2), : , :);
            bestcircle = bestMatch(1 : floor(height*1/4), : , :);
    
            if minD<70
            img_A = bestcircle;
            img_A = insertText(img_A, [bestcenters(1)-24,bestcenters(2)-34], num2str(i), 'TextColor', 'red', 'FontSize', 40, 'BoxOpacity', 0);
            img_A = insertShape(img_A, 'circle', bestcenters, 'Color', 'green', 'Opacity', 1);
    
            combinedImage = [img_A;bestfiber];
            outputImagePath = fullfile(outputFolder, sprintf('image_with_label_%d.png', i));
            imwrite(combinedImage, outputImagePath);
    
            circles = insertText(circles, [bestcenters(1)-24,bestcenters(2)-34], num2str(i), 'TextColor', 'red', 'FontSize', 40, 'BoxOpacity', 0);
            circles = insertShape(circles, 'circle',  [bestcenters(1),bestcenters(2),bestcenters(3)], 'Color', 'green', 'Opacity', 1);
            fiberPlace = insertShape(fiberPlace, 'FilledCircle', best_red_point, 'Color', 'red', 'Opacity', 1);    
            allData = [allData; i, [bestcenters(1), bestcenters(2)]];
            else
            count = count + 1;
            more(count) = i;    
            end
        end
    end
    
    more = more(1:count);
    if isempty(more)
    disp('empty');
    end
    
    for j = more
        minD = inf;
        bestMatch = []; % 存储最佳匹配的图像
        best_red_point = [];
        bestcenters = [];
        matches = struct('d', {}, 'dYR', {}, 'img', {}, 'redCenter', {}, 'centers', {});
    
        for k = 1:z
            % 获取红点数据
            redCenter = moreredPointData(k).Center_point;
            img = moreredPointData(k).Image;
            dYR = moreredPointData(k).length;
            centers = moreredPointData(k).Center_yellow;
    
            % 计算光纤末端和红点中心的距离
            if ~isempty(redCenter)
            d = sqrt((sortedF(j, 1) - redCenter(1)).^2 + (sortedF(j, 2) - redCenter(2)).^2);
            end
            %更新最小距离和最佳匹配
            if d < minD
                minD = d;
                bestMatch = img;
                best_red_point = redCenter;
                bestcenters = centers;            
            end
        end
         if ~isempty(bestMatch)
    
            [height, width] = size(bestMatch);       
            bestfiber = bestMatch(floor(height*3/4)+1 : end, : , :);
            bestcircle = bestMatch(1 : floor(height*1/4), : , :);
            bestpic = bestMatch(floor(height*1/4)+1 : floor(height*1/2), : , :);
            disp(minD);
    
            img_A = bestcircle;
            img_A = insertText(img_A, [bestcenters(1)-34,bestcenters(2)-34], num2str(j), 'TextColor', 'red', 'FontSize', 40, 'BoxOpacity', 0);
            img_A = insertShape(img_A, 'circle',bestcenters, 'Color', 'green', 'Opacity', 1);
    
            combinedImage = [img_A; bestpic;bestfiber];
            outputImagePath = fullfile(outputFolder, sprintf('image_with_label_%d.png', j));
            imwrite(combinedImage, outputImagePath);
    
            circles = insertText(circles, [bestcenters(1)-34,bestcenters(2)-34], num2str(j), 'TextColor', 'red', 'FontSize', 40, 'BoxOpacity', 0);
            circles = insertShape(circles, 'circle',[bestcenters(1),bestcenters(2),bestcenters(3)], 'Color', 'green', 'Opacity', 1);
            fiberPlace = insertShape(fiberPlace, 'FilledCircle', best_red_point, 'Color', 'red', 'Opacity', 1);
            allData = [allData; j, [bestcenters(1), bestcenters(2)]];
            % figure;
            % imshow(circles);
         end
    end
    % 保存全局图像
    %output1 = fullfile(outputFolder, sprintf('total_fibers.png'));
    %imwrite(fiberPlace, output1);
    output2 = fullfile(outputFolder, sprintf('total_circles.png'));
    imwrite(circles, output2);
    
    %% 
    %=====第三步 添加未识别项=====
    % 读取图像
    backgroundcircle = imread(fullfile(outputFolder, 'total_circles.png'));
    [height, width, ~] = size(backgroundcircle);
    
    % 提取 RGB 通道
    RR = backgroundcircle(:,:,1);
    GG = backgroundcircle(:,:,2);
    BB = backgroundcircle(:,:,3);
    
    brightness = rgb2gray(backgroundcircle);
    brightnessMask = brightness > 100;
    
    redMask = (RR > 200) & (GG < 50) & (BB < 50);     
    greenMask = (GG > 200) & (RR < 50) & (BB < 50);   
    [centersG, radiiG] = imfindcircles(greenMask, [30, 50], 'Sensitivity', 0.97);
    [x, y] = meshgrid(1:width, 1:height);
    
    circleMask = false(size(greenMask)); % 初始化掩膜
    for i = 1:length(radiiG)
        circleMask = circleMask | ((x - centersG(i,1)).^2 + (y - centersG(i,2)).^2 <= radiiG(i).^2);
    end
    outsideCircleMask = ~circleMask & ~redMask & brightnessMask;
    maskedarea = backgroundcircle .* uint8(outsideCircleMask);
    [centersB, radiiB] = imfindcircles(circlePlace,[25,38], 'Sensitivity', 0.97, 'EdgeThreshold', 0.1);
    centersC = [];
    radiiC = [];
    
    % 遍历所有 G 中的圆
    for i = 1:size(centersB, 1)
        mind = 1000;
        for j = 1:size(centersG, 1)
            % 计算 G 圆和 B 圆中心的距离
            d = sqrt((centersB(i,1) - centersG(j,1))^2 + (centersB(i,2) - centersG(j,2))^2);
            if d < mind
                mind = d;
            end
        end
       disp(mind);
       if mind > mean(radiiB)
           centersC = [centersC; centersB(i,:)];
           radiiC = [radiiC; radiiB(i)];
       end
    end
    
    % %（可选）在图像上绘制属于 C 的圆，确认未识别项
    % for j = 1:length(radiiC)
    %     maskedarea = insertShape(maskedarea, 'circle', [centersC(j,1), centersC(j,2), radiiC(j)], 'Color', 'yellow', 'Opacity', 1);
    % end
    % 
    % %显示结果
    % figure;
    % imshow(maskedarea);
    if ~isempty(centersC)
    l = sqrt((width - centersC(:,1)).^2+centersC(:,2).^2);
    [sortedl,idxl] = sort(l);
    sortedL = centersC(idxl,:);
    disp(length(sortedL));
    end
    
    for i = 1:length(invalidIndices)
        disp(i);
        disp(invalidIndices(i));
        circles = insertText(circles, [sortedL(i,1)-24,sortedL(i,2)-34], num2str(invalidIndices(i)), 'TextColor', 'red', 'FontSize', 40, 'BoxOpacity', 0);
        circles = insertShape(circles, 'circle',[sortedL(i,1),sortedL(i,2),radiiC(i)], 'Color', 'blue', 'Opacity', 1);
        allData = [allData; invalidIndices(i), [sortedL(i,1),sortedL(i,2)]];
    end    
    
    filename = 'output.xlsx';    % 写入文件
    writematrix(allData, filename, 'Sheet', 1, 'Range', 'A1');
    output2 = fullfile(outputFolder, sprintf('total_circles.png'));
    imwrite(circles, output2);
    
    
    matFilename = fullfile(outputFolder, 'fiber_registration_results.mat');
    
    registrationResults = struct();
    registrationResults.allData = allData;                    % 配准数据矩阵
    registrationResults.fiberCoordinates = sortedF;           % 光纤末端坐标
    registrationResults.circleCoordinates = centersC;         % 圆圈坐标
    registrationResults.invalidIndices = invalidIndices;      % 无效索引
    registrationResults.params = params;                      % 使用的参数
    registrationResults.timestamp = datetime('now');          % 处理时间
    
    registrationResults.columnDescriptions = {
        '第1列: 光纤编号', 
        '第2列: 输入端X坐标', 
        '第3列: 输入端Y坐标'
    };
   
    save(matFilename, 'registrationResults');

    disp('Matching complete.');
    
    fprintf('FC_4 处理完成，结果已保存到Excel文件\n');
end