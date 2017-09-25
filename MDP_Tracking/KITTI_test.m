% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% cross_validation on the KITTI benchmark
function KITTI_test(model, dataset)

    opt = globals();
    resume_from = 1;

    % load tracker from file
    tracker = load_tracker(opt.model_folder, model);

    num = numel(dataset.sequence_names);
    for j = resume_from:num
        fprintf('Testing on sequence: %s\n', dataset.sequence_names{j});
        MDP_test(j, 'testing', tracker, 1, model, dataset);
    end
end
