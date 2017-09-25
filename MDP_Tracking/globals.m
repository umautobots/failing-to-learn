% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
function opt = globals()

opt.root = erase(mfilename('fullpath'),'globals');

% path for tracking datasets
opt.kitti = [opt.root '../data/KITTI'];
opt.gta = [opt.root '../data/GTA'];

opt.kitti_train_seqs = {'0000', '0001', '0002', '0003', '0004', '0005', ...
    '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', ...
    '0015', '0016', '0017', '0018', '0019', '0020'};
opt.kitti_train_nums = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294, ...
    373, 78, 340, 106, 376, 209, 145, 339, 1059, 837];

opt.kitti_test_seqs = {'0000', '0001', '0002', '0003', '0004', '0005', ...
    '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', ...
    '0015', '0016', '0017', '0018', '0019', '0020', '0021', '0022', ...
    '0023', '0024', '0025', '0026', '0027', '0028'};
opt.kitti_test_nums = [465, 147, 243, 257, 421, 809, 114, 215, 165, 349, 1176, ...
    774, 694, 152, 850, 701, 510, 305, 180, 404, 173, 203, 436, 430, 316, 176, ...
    170, 85, 175];
opt.kitti_types = {'car', 'person', 'bicycle'};

opt.gta_test_seqs = arrayfun(@(n) sprintf('%04d',n), 0:103, 'UniformOutput',false);

opt.gta_test_nums = [ 851, 612, 448, 672, 706, 871, 416, 540, 890, 962, 747, 1054, ...
    835, 581, 471, 784, 458, 890, 936, 1091, 794, 455, 525, 1028, 970, 883, 583, 921, ...
    423, 979, 513, 1030, 1114, 161, 928, 924, 683, 912, 878, 512, 797, 571, 794, 876, ...
    880, 1057, 786, 374, 1055, 934, 390, 97, 966, 921, 405, 279, 842, 802, 790, 764, ...
    1047, 950, 355, 1056, 913, 751, 873, 642, 660, 824, 918, 1288, 1093, 926, 952, ...
    314, 637, 414, 607, 854, 597, 833, 635, 725, 898, 1123, 639, 937, 614, 954, 903, ...
    1101, 833, 945, 982, 1014, 540, 625, 1016, 955, 966, 689, 821, 825];

% addpath(fullfile(opt.mot, 'devkit', 'utils'));
addpath(fullfile(opt.kitti, 'devkit', 'matlab'));
addpath([opt.root '/3rd_party/libsvm-3.20/matlab']);
addpath([opt.root '/3rd_party/Hungarian']);

opt.results_kitti = fullfile(opt.root, 'results_kitti');
if exist(opt.results_kitti, 'dir') == 0
    mkdir(opt.results_kitti);
end
opt.model_folder = fullfile(opt.results_kitti, 'models');
if exist(opt.model_folder, 'dir') == 0
    mkdir(opt.model_folder);
end
opt.data_folder = fullfile(opt.results_kitti, 'data');
if exist(opt.data_folder, 'dir') == 0
    mkdir(opt.data_folder);
end

% tracking parameters
opt.num = 10;                 % number of templates in tracker (default 10)
opt.fb_factor = 30;           % normalization factor for forward-backward error in optical flow
opt.threshold_ratio = 0.6;    % aspect ratio threshold in target association
opt.threshold_dis = 3;        % distance threshold in target association, multiple of the width of target
opt.threshold_box = 0.8;      % bounding box overlap threshold in tracked state
opt.std_box = [30 60];        % [width height] of the stanford box in computing flow
opt.margin_box = [5, 2];      % [width height] of the margin in computing flow
opt.enlarge_box = [5, 3];     % enlarge the box before computing flow
opt.level_track = 1;          % LK level in association
opt.level =  1;               % LK level in association
opt.max_ratio = 0.9;          % min allowed ratio in LK
opt.min_vnorm = 0.2;          % min allowed velocity norm in LK
opt.overlap_box = 0.5;        % overlap with detection in LK
opt.patchsize = [24 12];      % patch size for target appearance
opt.weight_tracking = 1;      % weight for tracking box in tracked state
opt.weight_detection = 1;      % weight for detection box in tracked state
opt.weight_association = 1;   % weight for tracking box in lost state
opt.overlap_suppress1 = 0.5;   % overlap for suppressing detections with tracked objects
opt.overlap_suppress2 = 0.5;   % overlap for suppressing detections with tracked objects

% parameters for generating training data
opt.overlap_occ = 0.7;
opt.overlap_pos = 0.5;
opt.overlap_neg = 0.2;
opt.overlap_sup = 0.7;      % suppress target used in testing only

% training parameters
opt.max_iter = 10000;     % max iterations in total
opt.max_count = 10;       % max iterations per sequence
opt.max_pass = 2;

% parameters to transite to inactive
opt.max_occlusion = 50;
opt.exit_threshold = 0.95;
opt.tracked = 5;