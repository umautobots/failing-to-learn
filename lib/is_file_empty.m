% --------------------------------------------------------
% Failing to Learn
% Copyright (c) 2017 FCAV University of Michigan
% Licensed under The MIT License [see LICENSE for details]
% Written by Manikandasriram S.R. and Cyrus Anderson
% --------------------------------------------------------
function is_empty_flag = is_file_empty(filename)
    % skip empty ones
    temp = dir(filename);
    is_empty_flag = isempty(temp) || temp.bytes == 0;
end