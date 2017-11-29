% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
% This example script uses failing to learn algorithm to
% find false negatives given detections from a trained object
% detector using the temporal cue.
function test_failing_to_learn

	% add folders to path
	lib_path = erase(mfilename('fullpath'), 'test_failing_to_learn');
    addpath(lib_path);
    mdp_path = fullfile(lib_path, '..', 'MDP_Tracking');
    addpath(mdp_path);
    root_path = fullfile(lib_path, '..');

    % start logging
    logfile = fullfile(root_path, ['log_test_failing_to_learn_' datestr(datetime) '.txt']);
    diary(logfile);
    
    % check for input data
    % KITTI dataset, GTA dataset, detections from trained detector
    if ~exist(fullfile(root_path, 'data', 'KITTI', 'training', 'image_02'), 'dir') || ...
       ~exist(fullfile(root_path, 'data', 'KITTI', 'training', 'label_02'), 'dir')
        fprintf('ERROR:: KITTI dataset not found in <repo>/data/KITTI/training/.\n');
        return
    end
    if ~exist(fullfile(root_path, 'data', 'GTA', 'image_02'), 'dir') || ...
       ~exist(fullfile(root_path, 'data', 'GTA', 'label_02'), 'dir')
        fprintf('ERROR:: GTA dataset not found in <repo>/data/GTA/.\n');
        return
    end
    if ~exist(fullfile(root_path, 'results', 'KITTI', 'training', 'dets'), 'dir') || ...
       ~exist(fullfile(root_path, 'results', 'GTA', 'dets'), 'dir')
        fprintf('ERROR:: detections for KITTI or GTA datasets not found in <repo>/results/.\n');
        return
    end
    
    % Train forward MDP Tracker on KITTI dataset
    model = TrackerModel.fwd_kitti;
    dataset = Dataset.kitti_train;
    KITTI_train(model, dataset);
    
    % prepare backwards KITTI dataset
    dataset_bac = Dataset.kitti_train_backwards;
    if ~exist(dataset_bac.data_path, 'dir')
        prepare_backwards_data(Dataset.kitti_train, dataset_bac);
    end
    
    % Train backward MDP Tracker on KITTI dataset
    model = TrackerModel.bac_kitti;
    dataset = Dataset.kitti_train_backwards;
    KITTI_train(model, dataset);
    
    % Use trained forward MDP Tracker on GTA dataset
    model = TrackerModel.fwd_kitti;
    dataset = Dataset.gta;
    KITTI_test(model, dataset);
    
    % prepare backwards GTA dataset
    dataset_bac = Dataset.gta_backwards;
    if ~exist(dataset_bac.data_path, 'dir')
        prepare_backwards_data(Dataset.gta, dataset_bac);
    end
    
    % Use trained backward MDP Tracker on GTA dataset
    model = TrackerModel.bac_kitti;
    dataset = Dataset.gta_backwards;
    KITTI_test(model, dataset);
    
    % merge forward and backward tracks
    dataset = Dataset.gta;
    fwd_model = TrackerModel.fwd_kitti;
    dataset_bac = Dataset.gta_backwards;
    bac_model = TrackerModel.bac_kitti;
    model = TrackerModel.fwd_bac_kitti;
    merge_fwd_bac_tracks(fwd_model, dataset, bac_model, dataset_bac, model);
    
    % find false negatives
    dataset = Dataset.gta;
    dataset_bac = Dataset.gta_backwards;
    model = TrackerModel.fwd_bac_kitti;
    find_false_negatives(model, dataset, dataset_bac);
    
    % evaluate results
    dataset = Dataset.gta;
    PR_analysis(dataset, 0.7);
    
    % stop logging
    diary;
end
