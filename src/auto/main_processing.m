% main_processing.m
% Entry point for the automatic fiber localization and calibration pipeline.
% This script orchestrates data loading, parameter setup, ROI confirmation, and multi-stage calibration modules.
%
% Developed for fiber-array microscopy experiments.


function main_processing()

    clear; clc; close all;
    fprintf('开始光纤配准 ...\n');
    
    % 参数配置
    params = get_processing_parameters();

    % 初始化文件夹
    setup_folders(params);
    % outputFolder_fin = params.output_folder_fin;
   
        tic;
        fprintf('\n 提取关键帧...');
        [outputFolder_1, outputFolder_2] = FC_1_module(params);
        toc;tic;
        fprintf('\n 闪烁识别...');
        outputFolder_fin = FC_2_module(params, outputFolder_1, outputFolder_2);
        toc;        
        tic;
        fprintf('\n 初步配准质量检查...');
        invalidIndices = FC_3_module(params, outputFolder_fin);
        toc;
        invalidIndices = sort(invalidIndices);
        tic;
        fprintf('\n 生成最终配准结果...');
        FC_4_module(params, outputFolder_fin, invalidIndices);
        toc;tic;
        fprintf('\n 所有处理步骤完成.  \n');
    
    
end