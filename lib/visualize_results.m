% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------

function visualize_results
    close all;
    opt = globals();
    dataset = Dataset.gta;
    dataset_bac = Dataset.gta_backwards;
    fwd_model = TrackerModel.fwd_kitti;
    bac_model = TrackerModel.bac_kitti;
    model = TrackerModel.fwd_bac_kitti;
    
    resume_from = 1;
    only_if_found_fn = 0; % skip frames with no mistakes

    for i=resume_from:numel(dataset.sequence_names)
        seq_name = dataset.sequence_names{i};
        num_frames = dataset.sequence_lengths(i);

        has_gt = false;
        % Loading ground truth
        filename = fullfile(dataset.data_path, 'label_02', [seq_name '.txt']);
        if exist(filename, 'file')
            has_gt = true;
            fprintf('Processing kitti-format file: %s\n', filename);
            dres_gt = read_kitti2dres(filename, true);
        end

        % Load detections
        filename = fullfile(dataset.dets_path, [seq_name '.txt']);
        if ~exist(filename, 'file')
            fprintf('ERROR:: Detections not found for sequence %d at %s\n', i, filename);
            return
        end
        fprintf('Processing kitti-format file: %s\n', filename);
        if is_file_empty(filename)
            fprintf('Skipping seq %d since file %s is empty. Press any key to continue. \n', i, filename);
            pause();
            continue;
        end
        dres_det = read_kitti2dres(filename, false);
        
        % Load forward model
        filename = fullfile(opt.model_folder, fwd_model.save_folder, dataset.dataset_slug, [seq_name '.txt']);
        if ~exist(filename, 'file')
            fprintf('ERROR:: Detections not found for sequence %d at %s\n', i, filename);
            return
        end
        fprintf('Processing kitti-format file: %s\n', filename);
        if is_file_empty(filename)
            fprintf('Skipping seq %d since file %s is empty. Press any key to continue. \n', i, filename);
            pause();
            continue;
        end
        dres_fwd_tracks = read_kitti2dres(filename, true);

        % Load backward model
        filename = fullfile(opt.model_folder, bac_model.save_folder, dataset_bac.dataset_slug, [seq_name '.txt']);
        if ~exist(filename, 'file')
            fprintf('ERROR:: Detections not found for sequence %d at %s\n', i, filename);
            return
        end
        fprintf('Processing kitti-format file: %s\n', filename);
        if is_file_empty(filename)
            fprintf('Skipping seq %d since file %s is empty. Press any key to continue. \n', i, filename);
            pause();
            continue;
        end
        dres_bac_tracks = read_kitti2dres(filename, true);

        % Load forward backward model
        filename = fullfile(opt.model_folder, model.save_folder, [dataset.dataset_slug '_' dataset_bac.dataset_slug], [seq_name '.txt']);
        if ~exist(filename, 'file')
            fprintf('ERROR:: Detections not found for sequence %d at %s\n', i, filename);
            return
        end
        fprintf('Processing kitti-format file: %s\n', filename);
        if is_file_empty(filename)
            fprintf('Skipping seq %d since file %s is empty. Press any key to continue. \n', i, filename);
            pause();
            continue;
        end
        dres_fwd_bac_tracks = read_kitti2dres(filename, true);
        
        % Load false negatives
        filename = fullfile(dataset.false_negatives_path, 'temporal_cue', 'kitti_format', [seq_name '.txt']);
        fprintf('Processing kitti-format file: %s\n', filename);
        if is_file_empty(filename)
            fprintf('No mistakes found in seq %d. Press any key to continue to next seq. \n', i);
%             pause();
%             continue;
            dres_fn = struct('fr',{}, 'id',{},'type',cell(1),'x',{},'y',{},'w',{},'h',{},'r',{});
        else
            dres_fn = read_kitti2dres(filename, true);
        end


        % Show results
        for j=1:num_frames
            fr = j

            if only_if_found_fn && numel(find(dres_fn.fr==fr)) == 0
                continue;
            end
            
            image_file_name = sprintf('%06d.%s',fr-1, dataset.image_ext);
            image_path = fullfile(dataset.data_path, 'image_02', ...
                                seq_name, image_file_name);
            img = imread(image_path);

            % show ground truth results
            subplot(2, 3, 1);
            show_dres(fr, img, 'Ground Truth', dres_gt, 1, colormap, 0);

            % show detections
            subplot(2, 3, 2);
            show_dres(fr, img, 'Detections', dres_det, 1, colormap, 0);

            % show false negatives
            subplot(2, 3, 3);
            show_dres(fr, img, 'Found errors', dres_fn, 1, colormap, 0);
            
            % show forward tracks
            subplot(2, 3, 4);
            show_dres(fr, img, 'Fwd tracks', dres_fwd_tracks, 1, colormap, 0);
            
            % show backward tracks
            subplot(2, 3, 5);
            show_dres(num_frames-fr+1, img, 'Bac tracks', dres_bac_tracks, 1, colormap, 0);
            
            % show forward backward tracks
            subplot(2, 3, 6);
            show_dres(fr, img, 'Fwd bac tracks', dres_fwd_bac_tracks, 1, colormap, 0);
            

            pause();
        end
    end
end