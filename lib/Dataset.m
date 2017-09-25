% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
classdef Dataset
    properties
        sequence_names
        sequence_lengths
        data_path
        dets_path
        false_negatives_path
        dataset_slug
        image_ext
    end
    methods
        function dataset = Dataset(key)
            opt = globals();
            dataset.image_ext = 'png';
            if key == 1
                dataset.sequence_names = opt.kitti_train_seqs;
                dataset.sequence_lengths = opt.kitti_train_nums;
                dataset.data_path = fullfile(opt.kitti, 'training');
                dataset.dataset_slug = 'kitti';
                dataset.dets_path = fullfile(opt.root, '..', 'results', 'KITTI', 'training', 'dets');
                dataset.false_negatives_path = fullfile(opt.root, '..', 'results', 'KITTI', 'training', 'false_negatives');
            elseif key == 2
                dataset.sequence_names = opt.kitti_train_seqs;
                dataset.sequence_lengths = opt.kitti_train_nums;
                dataset.data_path = fullfile(opt.kitti, 'training', 'backwards');
                dataset.dataset_slug = 'kitti_backwards';
                dataset.dets_path = fullfile(opt.root, '..', 'results', 'KITTI', 'training', 'backwards', 'dets');
                dataset.false_negatives_path = fullfile(opt.root, '..', 'results', 'KITTI', 'training', 'backwards', 'false_negatives');
            elseif key == 3
                dataset.sequence_names = opt.kitti_test_seqs;
                dataset.sequence_lengths = opt.kitti_test_nums;
                dataset.data_path = fullfile(opt.kitti, 'testing');
                dataset.dataset_slug = 'kitti_test';
                dataset.dets_path = fullfile(opt.root, '..', 'results', 'KITTI', 'testing', 'dets');
                dataset.false_negatives_path = fullfile(opt.root, '..', 'results', 'KITTI', 'testing', 'false_negatives');
            elseif key == 4
                dataset.sequence_names = opt.kitti_test_seqs;
                dataset.sequence_lengths = opt.kitti_test_nums;
                dataset.data_path = fullfile(opt.kitti, 'testing', 'backwards');
                dataset.dataset_slug = 'kitti_test_backwards';
                dataset.dets_path = fullfile(opt.root, '..', 'results', 'KITTI', 'testing', 'backwards', 'dets');
                dataset.false_negatives_path = fullfile(opt.root, '..', 'results', 'KITTI', 'testing', 'backwards', 'false_negatives');
            elseif key == 5
                dataset.sequence_names = opt.gta_test_seqs;
                dataset.sequence_lengths = opt.gta_test_nums;
                dataset.data_path = opt.gta;
                dataset.dataset_slug = 'gta';
                dataset.image_ext = 'jpg';
                dataset.dets_path = fullfile(opt.root, '..', 'results', 'GTA', 'dets');
                dataset.false_negatives_path = fullfile(opt.root, '..', 'results', 'GTA', 'false_negatives');
            elseif key == 6
                dataset.sequence_names = opt.gta_test_seqs;
                dataset.sequence_lengths = opt.gta_test_nums;
                dataset.data_path = fullfile(opt.gta,'backwards');
                dataset.dataset_slug = 'gta_backwards';
                dataset.image_ext = 'jpg';
                dataset.dets_path = fullfile(opt.root, '..', 'results', 'GTA', 'backwards', 'dets');
                dataset.false_negatives_path = fullfile(opt.root, '..', 'results', 'GTA', 'backwards', 'false_negatives');
            end
        end
    end
    enumeration
        kitti_train           (1)
        kitti_train_backwards (2)
        kitti_test            (3)
        kitti_test_backwards  (4)
        gta                   (5)
        gta_backwards         (6)
    end
end
