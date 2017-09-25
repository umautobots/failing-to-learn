% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
classdef TrackerModel
    properties
        save_folder
    end
    methods
        function model = TrackerModel(save_folder)
            model.save_folder = save_folder;
        end
    end
    enumeration
        fwd_kitti     ('fwd-kitti')
        bac_kitti     ('bac-kitti')
        fwd_bac_kitti ('fwd_bac-kitti')
        fwd_gta       ('fwd-gta')
        bac_gta       ('bac-gta')
        fwd_bac_gta   ('fwd_bac-gta')
    end
end