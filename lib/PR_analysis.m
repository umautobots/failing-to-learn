% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
function PR_analysis(dataset, ov_thresh)

    opt = globals();

    % missed detections <=> GT has zero overlap with detections
    count_fn = zeros(numel(dataset.sequence_names),1);
    % false detections <=> detection has zero overlap with GT
    count_fp = zeros(numel(dataset.sequence_names),1);
    % true positives <=> detection has >70% overlap with GT
    count_tp = zeros(numel(dataset.sequence_names),1);
    % all detected objects
    count_total = zeros(numel(dataset.sequence_names),1);

    % detected error has non-zero overlap with GT
    count_err_tp = zeros(numel(dataset.sequence_names),1);
    % detected error has zero overlap with GT
    count_err_fp = zeros(numel(dataset.sequence_names),1);
    % detected error has zero overlap with false negatives
    count_err_fn = zeros(numel(dataset.sequence_names),1);
    % detected error is contained in Don't care region
    count_err_dc = zeros(numel(dataset.sequence_names),1);
    % total number of detected errors
    count_err_total = zeros(numel(dataset.sequence_names),1);

    hard_imgs = []; % images with non-zero false negatives
    err_imgs = []; % images with non-zero detected errors
    for i=1:numel(dataset.sequence_names)
        seq_name = dataset.sequence_names{i};
        num_frames = dataset.sequence_lengths(i);

        % Loading ground truth
        filename = fullfile(dataset.data_path, 'label_02', [seq_name '.txt']);
        dres_gt = read_kitti2dres(filename, true);
        
        % Load detections
        filename = fullfile(dataset.dets_path, [seq_name '.txt']);
        if is_file_empty(filename)
            fprintf('Skipping seq %d since file %s is empty\n', i, filename);
            continue;
        end
        dres_det = read_kitti2dres(filename, true);
        % filter for confidence
        dres_det = sub(dres_det, find(dres_det.r > 0.8));

        % Load errors
        filename = fullfile(dataset.false_negatives_path, 'temporal_cue', 'kitti_format', [seq_name '.txt']);

        if is_file_empty(filename)
            fprintf('Skipping seq %d since file %s is empty\n', i, filename);
            continue;
        end
        dres_errors = read_kitti2dres(filename, true);
        
        % Show results
        object_types = intersect(dres_gt.type, dres_errors.type);
        for j=1:num_frames
            object_type = 'car';
            fr = j;
            
            dc_inds = find(dres_gt.fr==fr & strcmp(dres_gt.type, 'DontCare'));
            gt_inds = find(dres_gt.fr==fr & strcmp(dres_gt.type, object_type));
            det_inds = find(dres_det.fr==fr & strcmp(dres_det.type, object_type));

            [dres_fn, tmp5, tmp6] = getFalseNegatives(dres_gt, gt_inds, dres_det, det_inds);
            if numel(dres_fn) > 0
                count_fn(i) = count_fn(i) + numel(dres_fn.id);
            end
            count_fp(i) = count_fp(i) + tmp5;
            count_tp(i) = count_tp(i) + tmp6;
            count_total(i) = count_total(i) + numel(det_inds);
            
            err_inds = find(dres_errors.fr==fr & strcmp(dres_errors.type, object_type));
            if numel(err_inds) == 0
                % no detected errors
                % might have to filter false negatives
                if numel(dres_fn) > 0
                    for k=1:numel(dres_fn.id)
                        ind_fn = k;
                        area_fn = dres_fn.w(ind_fn)*dres_fn.h(ind_fn);
                        if area_fn >= 1000
                            count_err_fn(i) = count_err_fn(i)+1;
                        end
                    end
                end
                tmp2 = 0;
            else
                [tmp1, tmp2, tmp3, tmp4] = evaluateErrors(dres_errors, err_inds, dres_gt, dc_inds, dres_fn, ov_thresh);
                count_err_tp(i) = count_err_tp(i) + tmp1;
                count_err_fp(i) = count_err_fp(i) + tmp2;
                count_err_fn(i) = count_err_fn(i) + tmp3;
                count_err_total(i) = count_err_total(i) + numel(err_inds)-tmp4;
            end
        end
    end
    fprintf('True Positives: %d, False Positives: %d, False Negatives: %d\n', sum(count_err_tp), sum(count_err_fp), sum(count_err_fn));
    fprintf('Precision: %d, Recall: %d\n', sum(count_err_tp)/sum(count_err_total), sum(count_err_tp)/(sum(count_err_tp)+sum(count_err_fn)));
    fprintf('Detector stats vv\n');
    fprintf('True Positives: %d, False Positives: %d, False Negatives: %d\n', sum(count_tp), sum(count_fp), sum(count_fn));
end


function [dres_fn, count_fp, count_tp] = getFalseNegatives(dres_gt, gt_inds, dres_det, det_inds)
    dres_fn = struct('fr', -99, 'id', -99, 'type', cell(1), ...
        'x', -99, 'y', -99, 'w', -99, 'h', -99, 'r', -99);
    if numel(det_inds) == 0
        % all GT are false negatives, no false positives
        fn_inds = 1:numel(gt_inds);
        count_tp = 0;
        count_fp = 0;
    elseif numel(gt_inds) == 0
        % no GT => no false negatives, all are false positives
        dres_fn = struct('fr', [], 'id', [], 'type', cell(0), ...
            'x', [], 'y', [], 'w', [], 'h', [], 'r', []);
        count_tp = 0;
        count_fp = numel(det_inds);
        return
    else
        costMatrix = zeros(numel(gt_inds), numel(det_inds));
        for k=1:numel(gt_inds)
            ind_gt = gt_inds(k);
            x1_gt = dres_gt.x(ind_gt);
            y1_gt = dres_gt.y(ind_gt);
            x2_gt = dres_gt.x(ind_gt) + dres_gt.w(ind_gt)-1;
            y2_gt = dres_gt.y(ind_gt) + dres_gt.h(ind_gt)-1;
            area_gt = dres_gt.w(ind_gt)*dres_gt.h(ind_gt);
            for l=1:numel(det_inds)
                ind_det = det_inds(l);
                x1_det = dres_det.x(ind_det);
                y1_det = dres_det.y(ind_det);
                x2_det = dres_det.x(ind_det) + dres_det.w(ind_det)-1;
                y2_det = dres_det.y(ind_det) + dres_det.h(ind_det)-1;
                area_det = dres_det.w(ind_det)*dres_det.h(ind_det);
                area_int = max(0,min(x2_det,x2_gt)-max(x1_det,x1_gt))*max(0,min(y2_det,y2_gt)-max(y1_det,y1_gt));
                ov = area_int/(area_gt+area_det-area_int);
                costMatrix(k,l) = 1/ov;
            end
        end
        [assignments, uG, uD] = assignDetectionsToTracks(costMatrix, 1.43);
        fn_inds = uG;
        count_tp = size(assignments,1);
        count_fp = numel(uD);
    end
    line_number = 1;
    for i=1:numel(fn_inds)
        ind_fn = gt_inds(fn_inds(i));
        dres_fn.id(line_number) = dres_gt.id(ind_fn);
        dres_fn.fr(line_number) = dres_gt.fr(ind_fn);
        dres_fn.type{line_number} = dres_gt.type{ind_fn};
        dres_fn.x(line_number) = dres_gt.x(ind_fn);
        dres_fn.y(line_number) = dres_gt.y(ind_fn);
        dres_fn.w(line_number) = dres_gt.w(ind_fn);
        dres_fn.h(line_number) = dres_gt.h(ind_fn);
        dres_fn.r(line_number) = dres_gt.r(ind_fn);
        line_number = line_number + 1;
    end
    if numel(fn_inds) == 0
        dres_fn = struct('fr', [], 'id', [], 'type', cell(0), ...
            'x', [], 'y', [], 'w', [], 'h', [], 'r', []);
    end
end

function [count_tp, count_fp, count_fn, count_dc_errors] = evaluateErrors(dres_err, err_inds, ...
                                        dres_dc, dc_inds, dres_fn, ov_thresh)
    % find errors not contained in dont care region
    count_dc_errors = 0;
    if numel(dc_inds) == 0
        valid_err_inds = err_inds;
    else
        valid_err_inds = [];
        for i=1:numel(err_inds)
            ind_err = err_inds(i);
            x1_err = dres_err.x(ind_err);
            y1_err = dres_err.y(ind_err);
            x2_err = dres_err.x(ind_err) + dres_err.w(ind_err)-1;
            y2_err = dres_err.y(ind_err) + dres_err.h(ind_err)-1;
            area_err = dres_err.w(ind_err)*dres_err.h(ind_err);
            inside_dc = 0;
            for j=1:numel(dc_inds)
                ind_dc = dc_inds(j);
                x1_dc = dres_dc.x(ind_dc);
                y1_dc = dres_dc.y(ind_dc);
                x2_dc = dres_dc.x(ind_dc) + dres_dc.w(ind_dc)-1;
                y2_dc = dres_dc.y(ind_dc) + dres_dc.h(ind_dc)-1;
                area_int = max(0,min(x2_dc,x2_err)-max(x1_dc,x1_err))*max(0,min(y2_dc,y2_err)-max(y1_dc,y1_err));
                if area_int > 0
                    % err is contained in dont care region
                    fprintf('Error overlaps DontCare region\n');
                    inside_dc = 1;
                    count_dc_errors = count_dc_errors + 1;
                    break;
                end
            end
            if inside_dc == 0
                valid_err_inds = [valid_err_inds, ind_err];
            end
        end
    end
    
    if numel(dres_fn) > 0
        costMatrix = zeros(numel(dres_fn.id), numel(valid_err_inds));
        for i=1:numel(dres_fn.id)
            ind_fn = i;
            x1_fn = dres_fn.x(ind_fn);
            y1_fn = dres_fn.y(ind_fn);
            x2_fn = dres_fn.x(ind_fn) + dres_fn.w(ind_fn)-1;
            y2_fn = dres_fn.y(ind_fn) + dres_fn.h(ind_fn)-1;
            area_fn = dres_fn.w(ind_fn)*dres_fn.h(ind_fn);
            for j=1:numel(valid_err_inds)
                ind_err = valid_err_inds(j);
                x1_err = dres_err.x(ind_err);
                y1_err = dres_err.y(ind_err);
                x2_err = dres_err.x(ind_err) + dres_err.w(ind_err)-1;
                y2_err = dres_err.y(ind_err) + dres_err.h(ind_err)-1;
                area_err = dres_err.w(ind_err)*dres_err.h(ind_err);
                area_int = max(0,min(x2_fn,x2_err)-max(x1_fn,x1_err))*max(0,min(y2_fn,y2_err)-max(y1_fn,y1_err));
                ov = area_int/(area_fn+area_err-area_int);
                costMatrix(i,j) = 1/ov;
            end
        end
        costMatrix;
        [assignments, uFN, uE] = assignDetectionsToTracks(costMatrix, 1/ov_thresh);
        count_tp = size(assignments,1);
        count_fp = numel(uE);
        count_fn = 0;
        for i=1:numel(uFN)
            ind_fn = uFN(i);
            area_fn = dres_fn.w(ind_fn)*dres_fn.h(ind_fn);
            if area_fn >= 1000
                count_fn = count_fn+1;
            end
        end
    else
        count_tp = 0;
        count_fp = numel(valid_err_inds);
        count_fn = 0;
    end
end

function s  = sub(s,I)
    % s = sub(s,I)
    % Returns a subset of the structure s

    if ~isempty(s)
      n = fieldnames(s);
      for i = 1:length(n)
        f = n{i};
        s.(f) = s.(f)(I,:);
      end
    end
end
