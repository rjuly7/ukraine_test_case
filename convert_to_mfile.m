clc, clear, close all;
load('test_case/ukraine_full.mat')
fname = 'data/full_limits.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);
% ids = fieldnames(val);
for i = 1:length(mpc.branch)
    %mpc.branch(i,6) = val.(ids{i});
    mpc.branch(i,6) = val(i,1);
end
savecase('test_case/ukraine_full.m',mpc);

% zbase = 220^2/(100);
% 
% R = 0.079535488
% X = 0.4721385909514598
% C = 10.374922529936367
% dist = 176.26117030808598
% 
% b = 2*60*pi*C*1e-9*zbase*dist