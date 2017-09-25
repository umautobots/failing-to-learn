% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
function dres_image = load_image_dres(save_path, seq_set, seq_name, seq_num, dataset)
	name = sprintf('kitti_%s_%s_dres_image.mat', seq_set, seq_name);
	filename = fullfile(save_path, dataset.dataset_slug, name);
	fprintf('Checking for cache %s\n', filename);

	% make the directories
	dirpath = fullfile(save_path, dataset.dataset_slug);
	if ~exist(dirpath, 'dir')
	    mkdir(dirpath);
	end

	if exist(filename, 'file') ~= 0
	    object = load(filename);
	    dres_image = object.dres_image;
	    fprintf('load images from file %s done\n', filename);
	else
	    dres_image = read_dres_image_kitti(dataset.data_path, seq_name, seq_num, dataset.image_ext);
	    fprintf('read images done\n');
	    save(filename, 'dres_image', '-v7.3');
	end
end