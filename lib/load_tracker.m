% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
function tracker = load_tracker(load_path, model, seq_name)
	if nargin < 3
		% if sequence is not mentioned, take the last one since it
		% is trained on more sequences
		foldername = fullfile(load_path, model.save_folder);
		files = dir([foldername '/*.mat']);
		name = files(end).name;
		filename = fullfile(load_path, model.save_folder, name);
	else
		name = sprintf('kitti_training_%s_tracker.mat', seq_name);
		filename = fullfile(load_path, model.save_folder, name);
	end
	object = load(filename);
	tracker = object.tracker;
	fprintf('load tracker from file %s\n', filename);
end