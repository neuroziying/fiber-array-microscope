function setup_folders(params)
    % 统一创建和管理所有输出文件夹
    
    folders = {params.output_folder_1, params.output_folder_2, ...
               params.output_folder_fin, params.output_folder_final};
    
    for i = 1:length(folders)
        if exist(folders{i}, 'dir')
            % 清空文件夹
            delete(fullfile(folders{i}, '*'));
        else
            % 创建文件夹
            mkdir(folders{i});
        end
    end
    fprintf('所有输出文件夹已准备就绪\n');
end