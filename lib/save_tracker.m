% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
function save_tracker(save_path, model, seq_name, tracker)
	name = sprintf('kitti_training_%s_tracker.mat', seq_name);
	filename = fullfile(save_path, model.save_folder, name);

	% make the directories
	dirpath = fullfile(save_path, model.save_folder);
	if ~exist(dirpath, 'dir')
	    mkdir(dirpath);
	end

	save(filename, 'tracker');
	fprintf('save tracker to file %s\n', filename);
end

