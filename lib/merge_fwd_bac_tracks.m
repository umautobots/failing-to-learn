% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
function merge_fwd_bac_tracks(fwd_model, dataset, bac_model, dataset_bac, model)
    opt = globals();

    num = numel(dataset.sequence_names);
    for i=1:num
        seq_name = dataset.sequence_names{i};
        num_frames = dataset.sequence_lengths(i);

        % read tracking results
        fwd_filename = fullfile(opt.model_folder, fwd_model.save_folder, dataset.dataset_slug, [seq_name '.txt']);
        bac_filename = fullfile(opt.model_folder, bac_model.save_folder, dataset_bac.dataset_slug, [seq_name '.txt']);
        if is_file_empty(fwd_filename) || is_file_empty(bac_filename)
            fprintf('skipping seq %d due to empty file: %s\n or %s\n', i, fwd_filename, bac_filename);
            continue;
        end
        fwd_dres_track = read_kitti2dres(fwd_filename, 1);
        bac_dres_track = read_kitti2dres(bac_filename, 1);
        
        write_dres = struct('fr', -99, 'id', -99, 'type', cell(1), ...
            'x', -99, 'y', -99, 'w', -99, 'h', -99, 'r', -99);
        line_number = 1;

        max_fwd_id = max(fwd_dres_track.id);
        % Each type is independently merged
        object_types = intersect(fwd_dres_track.type, bac_dres_track.type);
        for object_type_ind=1:numel(object_types)
            object_type = object_types{object_type_ind};
            fprintf('processing: %s\n', object_type);

            for j=1:num_frames
                fr = j;

                % splice dres to get fwd and bac tracks for current frame with object_type
                fwd_index_track = find(fwd_dres_track.fr == fr & ...
                    strcmp(fwd_dres_track.type, object_type));
                bac_index_track = find(bac_dres_track.fr == num_frames-fr+1 & ...
                    strcmp(bac_dres_track.type, object_type));

                if numel(fwd_index_track) == 0 && numel(bac_index_track) == 0
                    % no tracks maintained - go to next frame
                    fprintf('Found no tracks in %d. Continuing to next frame\n',fr);
                    continue;
                elseif numel(fwd_index_track) == 0
                    % only bac tracks are present - use all bac
                    for k=1:numel(bac_index_track)
                        bac_ind = bac_index_track(k);
                        write_dres.id(line_number) = max_fwd_id + bac_dres_track.id(bac_ind) + 1; % offset ID
                        write_dres.fr(line_number) = fr;
                        write_dres.type{line_number} = bac_dres_track.type{bac_ind};
                        write_dres.x(line_number) = bac_dres_track.x(bac_ind);
                        write_dres.y(line_number) = bac_dres_track.y(bac_ind);
                        write_dres.w(line_number) = bac_dres_track.w(bac_ind);
                        write_dres.h(line_number) = bac_dres_track.h(bac_ind);
                        write_dres.r(line_number) = bac_dres_track.r(bac_ind);
                        write_dres.state(line_number) = 2;  % hack->all items are 'tracked' and ok to write
                        line_number = line_number + 1;
                    end
                elseif numel(bac_index_track) == 0
                    % only fwd tracks are present - use all fwd
                    for k=1:numel(fwd_index_track)
                        fwd_ind = fwd_index_track(k);
                        write_dres.id(line_number) = fwd_dres_track.id(fwd_ind);
                        write_dres.fr(line_number) = fr;
                        write_dres.type{line_number} = fwd_dres_track.type{fwd_ind};
                        write_dres.x(line_number) = fwd_dres_track.x(fwd_ind);
                        write_dres.y(line_number) = fwd_dres_track.y(fwd_ind);
                        write_dres.w(line_number) = fwd_dres_track.w(fwd_ind);
                        write_dres.h(line_number) = fwd_dres_track.h(fwd_ind);
                        write_dres.r(line_number) = fwd_dres_track.r(fwd_ind);
                        write_dres.state(line_number) = 2;  % hack->all items are 'tracked' and ok to write
                        line_number = line_number + 1;
                    end
                else 
                    % use hungarian algorithm to associate fwd and bac tracks        
                    costMatrix = zeros(numel(fwd_index_track), numel(bac_index_track));
                    for k=1:numel(fwd_index_track)
                        fwd_ind = fwd_index_track(k);
                        x1_fwd = fwd_dres_track.x(fwd_ind);
                        y1_fwd = fwd_dres_track.y(fwd_ind);
                        x2_fwd = fwd_dres_track.x(fwd_ind) + fwd_dres_track.w(fwd_ind)-1;
                        y2_fwd = fwd_dres_track.y(fwd_ind) + fwd_dres_track.h(fwd_ind)-1;
                        area_fwd = fwd_dres_track.w(fwd_ind)*fwd_dres_track.h(fwd_ind);
                        for l=1:numel(bac_index_track)
                            bac_ind = bac_index_track(l);
                            x1_bac = bac_dres_track.x(bac_ind);
                            y1_bac = bac_dres_track.y(bac_ind);
                            x2_bac = bac_dres_track.x(bac_ind) + bac_dres_track.w(bac_ind)-1;
                            y2_bac = bac_dres_track.y(bac_ind) + bac_dres_track.h(bac_ind)-1;
                            area_bac = bac_dres_track.w(bac_ind)*bac_dres_track.h(bac_ind);
                            
                            area_int = max(0,min(x2_bac,x2_fwd)-max(x1_bac,x1_fwd))*max(0,min(y2_bac,y2_fwd)-max(y1_bac,y1_fwd));
                            ov = area_int/(area_bac+area_fwd-area_int);
                            costMatrix(k,l) = 1/ov;
                        end
                    end
                    % enforce >50% overlap
                    [assignments, uF, uB] = assignDetectionsToTracks(costMatrix, 2);
                    
                    % for associated pairs, choose the one with higher confidence
                    for k=1:size(assignments,1)
                        fwd_ind = fwd_index_track(assignments(k,1));
                        bac_ind = bac_index_track(assignments(k,2));
                        if fwd_dres_track.r(fwd_ind)>=bac_dres_track.r(bac_ind)
                            write_dres.id(line_number) = fwd_dres_track.id(fwd_ind);
                            write_dres.fr(line_number) = fr;
                            write_dres.type{line_number} = fwd_dres_track.type{fwd_ind};
                            write_dres.x(line_number) = fwd_dres_track.x(fwd_ind);
                            write_dres.y(line_number) = fwd_dres_track.y(fwd_ind);
                            write_dres.w(line_number) = fwd_dres_track.w(fwd_ind);
                            write_dres.h(line_number) = fwd_dres_track.h(fwd_ind);
                            write_dres.r(line_number) = fwd_dres_track.r(fwd_ind);
                            write_dres.state(line_number) = 2;  % hack->all items are 'tracked' and ok to write
                            line_number = line_number + 1;
                        else
                            write_dres.id(line_number) = max_fwd_id + bac_dres_track.id(bac_ind) + 1;
                            write_dres.fr(line_number) = fr;
                            write_dres.type{line_number} = bac_dres_track.type{bac_ind};
                            write_dres.x(line_number) = bac_dres_track.x(bac_ind);
                            write_dres.y(line_number) = bac_dres_track.y(bac_ind);
                            write_dres.w(line_number) = bac_dres_track.w(bac_ind);
                            write_dres.h(line_number) = bac_dres_track.h(bac_ind);
                            write_dres.r(line_number) = bac_dres_track.r(bac_ind);
                            write_dres.state(line_number) = 2;  % hack->all items are 'tracked' and ok to write
                            line_number = line_number + 1;
                        end
                    end

                    % add unassociated fwd tracks
                    for k=1:numel(uF)
                        fwd_ind = fwd_index_track(uF(k));
                        write_dres.id(line_number) = fwd_dres_track.id(fwd_ind);
                        write_dres.fr(line_number) = fr;
                        write_dres.type{line_number} = fwd_dres_track.type{fwd_ind};
                        write_dres.x(line_number) = fwd_dres_track.x(fwd_ind);
                        write_dres.y(line_number) = fwd_dres_track.y(fwd_ind);
                        write_dres.w(line_number) = fwd_dres_track.w(fwd_ind);
                        write_dres.h(line_number) = fwd_dres_track.h(fwd_ind);
                        write_dres.r(line_number) = fwd_dres_track.r(fwd_ind);
                        write_dres.state(line_number) = 2;  % hack->all items are 'tracked' and ok to write
                        line_number = line_number + 1;
                    end
                    % add unassociated bac tracks
                    for k=1:numel(uB)
                        bac_ind = bac_index_track(uB(k));
                        write_dres.id(line_number) = max_fwd_id + bac_dres_track.id(bac_ind) + 1;
                        write_dres.fr(line_number) = fr;
                        write_dres.type{line_number} = bac_dres_track.type{bac_ind};
                        write_dres.x(line_number) = bac_dres_track.x(bac_ind);
                        write_dres.y(line_number) = bac_dres_track.y(bac_ind);
                        write_dres.w(line_number) = bac_dres_track.w(bac_ind);
                        write_dres.h(line_number) = bac_dres_track.h(bac_ind);
                        write_dres.r(line_number) = bac_dres_track.r(bac_ind);
                        write_dres.state(line_number) = 2;  % hack->all items are 'tracked' and ok to write
                        line_number = line_number + 1;
                    end
                    fprintf('Processed %d-frame tracks\n',fr);
                end
            end
        end
        % write results to output file
        folder_path = fullfile(opt.model_folder, model.save_folder);
        if ~exist(folder_path, 'dir')
            mkdir(folder_path);
        end
        folder_path = fullfile(folder_path, [dataset.dataset_slug '_' dataset_bac.dataset_slug]);
        if ~exist(folder_path, 'dir')
            mkdir(folder_path);
        end
        name = sprintf('%s.txt', seq_name);
        filename = fullfile(folder_path, name);
        fprintf('write results: %s\n', filename);
        opt.tracked = 0;  % deb
        write_tracking_results_kitti(filename, write_dres, opt.tracked);
end
