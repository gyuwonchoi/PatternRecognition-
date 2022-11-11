clear;

% load data
trainset = load_data('./regressor_trn.txt');
testset = load_data('./regressor_tst.txt');

% data param
data_num = length(trainset); % sample num 

% train param 
node_num = 10;                % num of nodes in hidden layer 
input_num = 2+1;             % x1, x2, and bias -1
epoch = 50000;
lr = 0.0000001;                   % learning rate?

% input initialization 
input = trainset(:, 1:2);
bias  = -ones(data_num, 1);
input = [input  bias];

dk = trainset(:, 3);

% weights declariation 
wj = normrnd(0, 1, [input_num, node_num])    % node_num x 3
wk = normrnd(0, 1, [node_num, 1]);           % 3x1

x = 0
y = -inf
% train : Neuron j & k 
for i= 1:epoch
    % forward
    vj = input * wj;           % neuron j
    yj = sigmoid(vj);

    vk = yj * wk;              % neuron k
    yk = sigmoid(vk);

    error = dk - yk;         
    error_batch = diag(error * error.');
    error_sum = sum(error_batch, 1);

    % backward 
    delta_k = error * yk.'* (1-yk);     % neuron k
    wght_del_k = lr * delta_k.' * yj;   % batch mode 
    size(yj)

    % delta_j_sig = sum(delta_k * wk.',2); % each neuron(6) in each sample
    delta_j_sig = delta_k * wk.'; % 225 x 6 
    delta_j = diag(yj* (1-yj.'))+ delta_j_sig; % 225 x 6

    % wght_del_j = lr * delta_j.' * input;        
    wght_del_j = lr * input.' * delta_j; % 3x6

    % update weight 
    wj = wj + wght_del_j;
    wk = wk + wght_del_k.';
    
    size(wght_del_k)
    % draw chart error_sum x epoch
    y = [y error_sum];
    x = [x i];
end

plot(x,y)

function data = sigmoid_prime(output)
    data = output.* (1 -output);
end 

function mat = load_data(path)
    fid = fopen(path);
    for i=1:4, buffer = fgetl(fid); end % remove headers

    data = textscan(fid, '%f%f%f', 'Delimiter','\t');
    
    mat= zeros(225,3);
    mat(:, 1)= data{1,1};
    mat(:, 2)= data{1,2};
    mat(:, 3)= data{1,3};
    
    fclose(fid);
end


