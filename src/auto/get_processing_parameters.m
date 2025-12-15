function params = get_processing_parameters()
    % 统一管理所有程序的参数
    
    params = struct();
    
    %% 选择数据文件夹

    params.data_folder = uigetdir('.', '请选择数据文件夹');

    [~, folderName] = fileparts(params.data_folder);
    params.output_folder_1 = [folderName '-start'];
    params.output_folder_2 = [folderName '-bkg'];
    params.output_folder_fin = [folderName '-fin']; 
    params.output_folder_final = [folderName '-output'];

    %% ROI参数确认
    params.fc1.video_file = 'HDFP1.AVI';           % 视频名称
    params.fc1.circle_image = 'up1.jpg';           % 光纤进光端模板图（圆圈更大的一端）    ****一定要修改文件名，不然找不到目标文件，读取会报错***
    params.fc2.fiber_image = 'down1.jpg';          % 光纤出光端模板图（圆圈更小的一端）



    circle_image = imread(fullfile(params.data_folder, params.fc1.circle_image));  % 读取光纤图像（进光端/圆圈半径较大的一端）
    fiber_image = imread(fullfile(params.data_folder, params.fc2.fiber_image));

    [params.fc1.left, params.fc1.right, params.fc1.up, params.fc1.down] = confirm_roi_parameters(circle_image, '光纤输入端', [50, 50, 50, 50]);
    [params.fc2.left, params.fc2.right, params.fc2.up, params.fc2.down] = confirm_roi_parameters(fiber_image, '光纤输出端', [50, 50, 50, 50]);

    
    %% FC_1 参数

    % params.fc1.left = 50;      %剪裁进光端模板ROI区域。 E.g.如果希望左侧裁掉50像素的无效区域，则设置left = 50
    % params.fc1.right = 50;
    % params.fc1.up = 50;
    % params.fc1.down = 50;

    params.fc1.LI = 50;        %剪裁视频ROI区域。 E.g.如果希望左侧裁掉600像素的无效区域，则设置left = 600
    params.fc1.RI = 50;

    params.fc1.brightness_threshold_L = 60;         % 最低亮度阈值：判定是否存在光照。一般无需更改。（20~60）
    params.fc1.brightness_threshold_H = 250;        % 最高亮度阈值：光纤被照射后亮起的最大亮度。一般无需更改 （230~255）
    params.fc1.threshold = 20;                      % 重叠判定：光圈与光纤圆心间最短距离（pix）。一般无需更改

    
    %% FC_2 参数  

    % params.fc2.left = 50;       %剪裁出光端模板ROI区域。 E.g.如果希望左侧裁掉300像素的无效区域，则设置left = 300
    % params.fc2.right = 50;
    % params.fc2.up = 50;
    % params.fc2.down = 50;

    params.fc2.brightness_threshold = 60;         % 亮度阈值：判断是否有闪烁出现。参考范围（30，80），当闪烁较明显时取>50，闪烁暗淡时取<30
    
    %% FC_3 和 FC_4 共用参数

    params.fc3.left = params.fc2.left;
    params.fc3.right = params.fc2.right; 
    params.fc3.up = params.fc2.up;
    params.fc3.down = params.fc2.down;

    params.fc3.circle_image = params.fc1.circle_image;   
    params.fc3.fiber_image =  params.fc2.fiber_image ;

    params.fc3.brightness_threshold = 200;          %亮度阈值，取在（120，250）之间，判断光纤是否被对应光斑点亮。一般无需更改。
                                                    % 当视频比较暗淡时，适当调低阈值。
                                                    % 当检测到的亮点数过多（>10k)时，适当调高阈值
    
                                                    
    params.fc3.spot_count_threshold = 400;          % 检查点数和范围，直接关系到配准结果正确性。取值最好为 '找到亮点数量'的中位数或更低。
                                                    % 这里没有增加交互功能，经验来看取值范围为（400，1500），质量较好的视频一般取（400~800）。
                                                    % 
                                                    % 判断取值是否合理的方法：
                                                    % 【1】 一半以上的'亮点数量'都大于阈值
                                                    % 【2】 FC_3 的运行结果配准效果较好。若重叠的标签很多，说明这里的值取大了。最好调整为 [ 重叠数量<5 个 ] 的状态。
                                                    % 
    
end