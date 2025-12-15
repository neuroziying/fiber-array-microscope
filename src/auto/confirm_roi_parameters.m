function [left, right, up, down] = confirm_roi_parameters(image, roi_name, default_params)
    % 交互式确认ROI参数
    % 输入: image - 图像数据
    %       roi_name - ROI区域名称
    %       default_params - 默认参数 [left, right, up, down]
    % 输出: 确认后的ROI参数
    
    [height, width, ~] = size(image);
    
    
    % 显示默认ROI
    default_roi = image(default_params(3)+1:height-default_params(4),default_params(1)+1:width-default_params(2), :);
    figure('Position', [200, 200, 800, 600]);
    imshow(default_roi);
    title(sprintf('%s - 默认ROI', roi_name));
    
  
    modify = input('是否修改ROI参数？(y/n) [n]: ', 's');
    
    if strcmpi(modify, 'y')

        left = input(sprintf('输入left值 [默认: %d]: ', default_params(1)));
        if isempty(left), left = default_params(1); end
        
        right = input(sprintf('输入right值 [默认: %d]: ', default_params(2)));
        if isempty(right), right = default_params(2); end
        
        up = input(sprintf('输入up值 [默认: %d]: ', default_params(3)));
        if isempty(up), up = default_params(3); end
        
        down = input(sprintf('输入down值 [默认: %d]: ', default_params(4)));
        if isempty(down), down = default_params(4); end
        
        
        if left < 0 || right < 0 || up < 0 || down < 0
            error('ROI参数不能为负数');
        end
        if (left + right) >= width || (up + down) >= height
            error('ROI参数过大，裁剪后无有效区域');
        end
        
        % 确认新ROI
        new_roi = image(up+1:height-down, left+1:width-right, :);
        figure('Position', [200, 200, 800, 600]);
        imshow(new_roi);
        title(sprintf('%s - 新ROI: left=%d, right=%d, up=%d, down=%d', roi_name, left, right, up, down));
        
        confirm = input('确认使用这个ROI？(y/n) [y]: ', 's');
        if strcmpi(confirm, 'n')
            [left, right, up, down] = confirm_roi_parameters(image, roi_name, [left, right, up, down]);
        end

    else
        left = default_params(1);
        right = default_params(2);
        up = default_params(3);
        down = default_params(4);
        fprintf('使用默认ROI参数\n');
    end
    close all;
    
    fprintf('%s ROI确认完成: left=%d, right=%d, up=%d, down=%d\n', roi_name, left, right, up, down);
end