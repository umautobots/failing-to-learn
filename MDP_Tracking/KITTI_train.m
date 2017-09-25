% Adapted by Manikandasriram S.R. and Cyrus Anderson for Failing to Learn
% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% train MDP Tracker on KITTI-like dataset
function KITTI_train(model, dataset)

opt = globals();
is_loading_from = -1;  % for training: index of tracker in seq_idx, or <1 for no-load

% number of training sequences
num = numel(dataset.sequence_names);

tracker = [];
% online training
for i = 1:num
    
    if is_loading_from > 0
        if i < is_loading_from
            continue
        end
        fprintf('Loading tracker on kitti: %d\n', i); 
        tracker = load_tracker(opt.model_folder, model, dataset.sequence_names{i});
        is_loading_from = -1;  % do not load again
        continue
    end
    
    fprintf('Online training on sequence: %s\n', dataset.sequence_names{i});
    % Check if detections are available
    filename = fullfile(dataset.dets_path, [dataset.sequence_names{i} '.txt']);
    if ~exist(filename, 'file')
        disp('ERROR::Missing detections. Aborting.');
        return;
    end
    tracker = MDP_train(i, tracker, 1, model, dataset);
end
fprintf('%d training examples after online training\n', size(tracker.f_occluded, 1));