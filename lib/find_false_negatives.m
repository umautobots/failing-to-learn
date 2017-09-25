% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
function find_false_negatives(model, dataset, dataset_bac)

    opt = globals();

    output_folder = fullfile(dataset.false_negatives_path, 'temporal_cue');
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
        mkdir(fullfile(output_folder,'JPEGImages'));
        mkdir(fullfile(output_folder,'Annotations'));
        mkdir(fullfile(output_folder,'kitti_format'));
    end

    num = numel(dataset.sequence_names);
    for i=1:num
    
        seq_name = dataset.sequence_names{i};
        num_frames = dataset.sequence_lengths(i);
    
        % read detections
        filename = fullfile(dataset.dets_path, [seq_name '.txt']);
        fprintf('Processing kitti-format file: %s\n', filename)
        if is_file_empty(filename)
            fprintf('Skipping %d since file %s is empty\n', i, filename);
            continue;
        end
        dres_det = read_kitti2dres(filename);

        % read merged tracks
        filename = fullfile(opt.model_folder, model.save_folder, [dataset.dataset_slug '_' dataset_bac.dataset_slug], [seq_name '.txt']);
        fprintf('Processing kitti-format file: %s\n', filename)
        if is_file_empty(filename)
            fprintf('Skipping %d since file %s is empty\n', i, filename);
            continue;
        end
        dres_track = read_kitti2dres(filename);

        write_dres = struct('fr', -99, 'id', -99, 'type', cell(1), ...
                            'x', -99, 'y', -99, 'w', -99, 'h', -99, 'r', -99);
        line_number = 1;
        rec = struct();
        frame2object_num = zeros(num_frames, 1);

        % process each type separately
        object_types = intersect(dres_det.type, dres_track.type);
        for object_type_ind=1:numel(object_types)
            object_type = object_types{object_type_ind};
            fprintf('processing: %s\n', object_type);
            
            for frame_i=1:num_frames
                det_inds = find(dres_det.fr == frame_i & ...
                    strcmp(dres_det.type, object_type));
                track_inds = find(dres_track.fr == frame_i & ...
                    strcmp(dres_track.type, object_type));

                % skip if no tracks are present
                if numel(track_inds) == 0
                    continue;
                elseif numel(det_inds) == 0
                    % no detections - use all tracks
                    
                    % remove if tracks overlap with each other
                    % valid_track_inds = prune_overlapping_tracks(dres_track, track_inds);

                    valid_track_inds = track_inds;
                else
                    [~, uT, uD, costMatrix] = findAssignment(dres_det, dres_track, det_inds, track_inds);

                    % skip if no unassociated tracks
                    if numel(uT) == 0
                        continue;
                    end

                    % find unassociated tracks with no overlapping detections
                    % since cost matrix is already computed, if the rows corresponding
                    % to uT are filled with only Inf, they dont overlap any detections
                    subMatrix = costMatrix(uT,:);
                    valid_uTs = find(min(subMatrix,[],2)==Inf);
                    if numel(valid_uTs) == 0
                       % skip since no rows with only Inf exist
                       continue; 
                    end

                    % remove if uTs overlap with each other
                    % valid_track_inds = prune_overlapping_tracks(dres_track, track_inds(uT(valid_uTs)));

                    % no removal of the overlapping uTs
                    valid_track_inds = track_inds(uT(valid_uTs));
                end
                
                if numel(valid_track_inds) == 0
                    % no non-overlapping uT
                    continue;
                end
                
                % load image information
                image_file_name = sprintf('%06d.%s',frame_i-1, dataset.image_ext);
                original_image_path = fullfile(dataset.data_path, 'image_02', ...
                                                seq_name, image_file_name);
                img_info = imfinfo(original_image_path);

                for l=1:numel(valid_track_inds)
                    track_i = valid_track_inds(l);
                    x1_track = dres_track.x(track_i);
                    y1_track = dres_track.y(track_i);
                    x2_track = x1_track + dres_track.w(track_i)-1;
                    y2_track = y1_track + dres_track.h(track_i)-1;

                    % removing small bbox
                    if is_bb_small(x1_track, y1_track, x2_track, y2_track)
                        continue;
                    end

                    % removing boxes near edges
                    if is_edge_track(x1_track, y1_track, x2_track, y2_track, img_info.Width, img_info.Height)
                        continue;
                    end
                    
                    % write to record
                    if frame2object_num(frame_i) == 0
                        rec(frame_i).annotation.folder = output_folder;
                        rec(frame_i).annotation.filename = image_file_name;
                        rec(frame_i).annotation.source.database = dataset.dataset_slug;
                        rec(frame_i).annotation.source.annotation = 'Failing to Learn';
                        rec(frame_i).annotation.source.raw_image = image_file_name;
                        rec(frame_i).annotation.size.width = img_info.Width;
                        rec(frame_i).annotation.size.height = img_info.Height;
                        rec(frame_i).annotation.size.depth = 3;
                        rec(frame_i).annotation.segmented = '0';
                        rec(frame_i).annotation.source.image = image_file_name;
                    end
                    o_i = frame2object_num(frame_i) + 1;
                    rec(frame_i).annotation.object(o_i).name = dres_track.type{track_i};
                    rec(frame_i).annotation.object(o_i).pose = 'Unspecified';
                    rec(frame_i).annotation.object(o_i).truncation = '0';
                    rec(frame_i).annotation.object(o_i).difficult = '-1';
                    rec(frame_i).annotation.object(o_i).bndbox.xmin = x1_track;
                    rec(frame_i).annotation.object(o_i).bndbox.ymin = y1_track;
                    rec(frame_i).annotation.object(o_i).bndbox.xmax = x2_track;
                    rec(frame_i).annotation.object(o_i).bndbox.ymax = y2_track;
                    rec(frame_i).annotation.object(o_i).id = dres_track.id(track_i);
                    rec(frame_i).annotation.object(o_i).score = dres_track.r(track_i);
                    frame2object_num(frame_i) = o_i;

                    write_dres.id(line_number) = dres_track.id(track_i);
                    write_dres.fr(line_number) = dres_track.fr(track_i);
                    write_dres.type{line_number} = dres_track.type{track_i};
                    write_dres.x(line_number) = dres_track.x(track_i);
                    write_dres.y(line_number) = dres_track.y(track_i);
                    write_dres.w(line_number) = dres_track.w(track_i);
                    write_dres.h(line_number) = dres_track.h(track_i);
                    write_dres.r(line_number) = dres_track.r(track_i);
                    write_dres.state(line_number) = 2;  % hack->all items are 'tracked' and ok to write
                    line_number = line_number + 1;
                end
                fprintf('processed %s\n', original_image_path);
            end
        end  % object types
    
        % write image, xml
        for frame_i=1:num_frames
            if frame2object_num(frame_i) == 0
                % no mistakes found in this frame
                continue;
            end
            image_file_name = sprintf('%06d.%s',frame_i-1, dataset.image_ext);
            original_image_path = fullfile(dataset.data_path, 'image_02', ...
                                           seq_name, image_file_name);
            if ~exist(fullfile(output_folder, 'JPEGImages', seq_name), 'dir')
                mkdir(fullfile(output_folder, 'JPEGImages', seq_name));
            end
            new_image_path = fullfile(output_folder, 'JPEGImages', seq_name, image_file_name);

            fprintf('copying %s to %s\n', original_image_path, new_image_path);
            copyfile(original_image_path,new_image_path);
            
            xml_file_name = sprintf('%06d.xml', frame_i-1);
            if ~exist(fullfile(output_folder, 'Annotations', seq_name), 'dir')
                mkdir(fullfile(output_folder, 'Annotations', seq_name));
            end
            new_xml_path = fullfile(output_folder, 'Annotations', seq_name, xml_file_name);
            fprintf('writing %s\n', new_xml_path);
            VOCwritexml(rec(frame_i), new_xml_path);
        end
        % write results to output file
        name = sprintf('%s.txt', seq_name);
        filename = fullfile(output_folder, 'kitti_format', name);
        fprintf('write results: %s\n', filename);
        opt.tracked = 0;  % deb
        if write_dres.id(1) == -99
            % nothing to write
            fclose(fopen(filename, 'w'));
        else
            write_tracking_results_kitti(filename, write_dres, opt.tracked);
        end
    end

end

% Constructs a cost matrix with inverse of bbox IoUs as cost for tracks vs dets
% use Hungarian algorithm to assign tracks to dets with IoU <=50% left unassigned
function [assignments, uT, uD, costMatrix] = findAssignment(dres_det, dres_track, det_inds, track_inds)
    costMatrix = zeros(numel(track_inds), numel(det_inds));
    for k=1:numel(track_inds)
        track_ind = track_inds(k);
        x1_track = dres_track.x(track_ind);
        y1_track = dres_track.y(track_ind);
        x2_track = x1_track + dres_track.w(track_ind)-1;
        y2_track = y1_track + dres_track.h(track_ind)-1;
        area_track = dres_track.w(track_ind)*dres_track.h(track_ind);
        for l=1:numel(det_inds)
            det_ind = det_inds(l);
            x1_det = dres_det.x(det_ind);
            y1_det = dres_det.y(det_ind);
            x2_det = x1_det + dres_det.w(det_ind)-1;
            y2_det = y1_det + dres_det.h(det_ind)-1;
            area_det = dres_det.w(det_ind)*dres_det.h(det_ind);
            area_int = max(0,min(x2_det,x2_track)-max(x1_det,x1_track))*max(0,min(y2_det,y2_track)-max(y1_det,y1_track));
            ov = area_int/(area_det+area_track-area_int);
            costMatrix(k,l) = 1/ov;
        end
    end
    [assignments, uT, uD] = assignDetectionsToTracks(costMatrix, 2);
end

% Checks if any tracks overlap with each other
% returns non-overlapping track indices
function non_overlapping_inds = prune_overlapping_tracks(dres_track, track_inds)
    overlapping_bb = zeros(numel(track_inds), 1);
    for i=1:numel(track_inds)-1
        x1 = dres_track.x(track_inds(i));
        y1 = dres_track.y(track_inds(i));
        w1 = dres_track.w(track_inds(i));
        h1 = dres_track.h(track_inds(i));
        for j=i+1:numel(track_inds)
            ox1 = dres_track.x(track_inds(j));
            oy1 = dres_track.y(track_inds(j));
            ow1 = dres_track.w(track_inds(j));
            oh1 = dres_track.h(track_inds(j));
            overlap = bboxOverlapRatio([ox1, oy1, ow1, oh1], [x1, y1, w1, h1]);
            if overlap
                overlapping_bb(i) = 1;
                overlapping_bb(j) = 1;
            end
        end
    end
    non_overlapping_inds = track_inds(~overlapping_bb);
end

% checks if bbox is smaller than a threshold
function flag = is_bb_small(x1, y1, x2, y2)
    min_area = 1000;
    flag = 0;
    if (x2-x1)*(y2-y1) < min_area
        flag = 1;
    end
end

% checks if bbox is closer than a threshold to the image edges
function flag = is_edge_track(x1, y1, x2, y2, img_width, img_height)
    min_w_padding = 20;
    min_h_padding = 20;
    flag = x1 < min_w_padding;
    flag = flag || y1 < min_h_padding;
    flag = flag || img_width - x2 < min_w_padding;
    flag = flag || img_height - y2 < min_h_padding;
end
