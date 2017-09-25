% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
function prepare_backwards_data(dataset, dataset_bac, skip_gt)

    if nargin < 3
        skip_gt = false;
    end

    if ~skip_gt
        source_folder_name = fullfile(dataset.data_path, 'label_02');
        src_image_folder = fullfile(dataset.data_path, 'image_02');
        dest_folder_name = fullfile(dataset_bac.data_path, 'label_02');
        dest_image_folder = fullfile(dataset_bac.data_path, 'image_02');
        
        if ~exist(dest_folder_name, 'dir')
            mkdir(dest_folder_name);
        end
        if ~exist(dest_image_folder, 'dir')
            mkdir(dest_image_folder);
        end

        fprintf('Preparing ground truth labels\n');

        for j=1:numel(dataset.sequence_names)
            src_file = fullfile(source_folder_name, [dataset.sequence_names{j} '.txt']);
            fprintf('Processing kitti-format file: %s\n', src_file)
            dres = read_kitti2dres(src_file, 1);
            num_frames = dataset.sequence_lengths(j);

            dest_file = fullfile(dest_folder_name, [dataset.sequence_names{j} '.txt']);
            fprintf('Writing reverse kitti-format file: %s\n', dest_file)
            fid = fopen(dest_file, 'w');
            
            N = numel(dres.x);
            rev_ids = flipud(dres.id);
            uniq_ids = unique(rev_ids,'stable');
            % since it is in reverse, also reverse the order so frame 1 is at the
            % top (otherwise MDP training breaks)
            for i = N:-1:1
                % <frame>, <id>, <type>, <truncated>, <occluded>, <alpha>, 
                % <bb_left>, <bb_top>, <bb_right>, <bb_bottom>, <3D height>, <3D width>, <3D length>
                % <3D x>, <3D y>, <3D z>, <rotation y>, <conf>
                % subtract extra -1 from id, x+h, y+h due to the MDP loader adding +1
                fprintf(fid, '%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f\n', ...
                    num_frames-dres.fr(i), find(uniq_ids==dres.id(i)), dres.type{i}, -1, -1, -1, ...
                    dres.x(i), dres.y(i), dres.x(i)+dres.w(i)-1, dres.y(i)+dres.h(i)-1, ...
                    -1, -1, -1, -1, -1, -1, -1);
            end
            fclose(fid);
            
            % Copy over images
            if ~exist(fullfile(dest_image_folder, dataset.sequence_names{j}), 'dir')
                mkdir(fullfile(dest_image_folder, dataset.sequence_names{j}));
            end
            for i=1:num_frames    
                src_image_file = fullfile(src_image_folder, dataset.sequence_names{j}, sprintf('%06d.%s', i-1, dataset.image_ext));
                dest_image_file = fullfile(dest_image_folder, dataset.sequence_names{j}, sprintf('%06d.%s', num_frames-i, dataset.image_ext));
                fprintf('Copying from %s to %s\n', src_image_file, dest_image_file);
                copyfile(src_image_file, dest_image_file);
            end
        end
    end

    fprintf('Preparing detections\n');

    source_folder_name = dataset.dets_path;
    dest_folder_name = dataset_bac.dets_path;

    if ~exist(dest_folder_name, 'dir')
        mkdir(dest_folder_name);
    end

    for i=1:numel(dataset.sequence_names)
        src_file = fullfile(source_folder_name, [dataset.sequence_names{i} '.txt']);
        fprintf('Processing kitti-format file: %s\n', src_file)
        dres = read_kitti2dres(src_file, 1);
        num_frames = dataset.sequence_lengths(i);

        dest_file = fullfile(dest_folder_name, [dataset.sequence_names{i} '.txt']);
        fprintf('Writing reverse kitti-format file: %s\n', dest_file)
        fid = fopen(dest_file, 'w');
        
        N = numel(dres.x);
        rev_ids = flipud(dres.id);
        uniq_ids = unique(rev_ids,'stable');
        % since it is in reverse, also reverse the order so frame 1 is at the
        % top (otherwise MDP training breaks)
        for i = N:-1:1
            % <frame>, <id>, <type>, <truncated>, <occluded>, <alpha>, 
            % <bb_left>, <bb_top>, <bb_right>, <bb_bottom>, <3D height>, <3D width>, <3D length>
            % <3D x>, <3D y>, <3D z>, <rotation y>, <conf>
            % subtract extra -1 from id, x+h, y+h due to the MDP loader adding +1
            cf = dres.r(i);
            fprintf(fid, '%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f\n', ...
                num_frames-dres.fr(i), -1, dres.type{i}, -1, -1, -1, ...
                dres.x(i), dres.y(i), dres.x(i)+dres.w(i)-1, dres.y(i)+dres.h(i)-1, ...
                -1, -1, -1, -1, -1, -1, -1, cf);
        end
        fclose(fid); 
    end
end