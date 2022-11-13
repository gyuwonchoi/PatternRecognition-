clear;

% load data
trainset = load_data('./regressor_trn.txt');
testset = load_data('./regressor_tst.txt');

% data param
data_num = length(trainset);  % sample num 

% train param 
node_num = 10;                % num of nodes in hidden layer 
input_num = 2+1;              % x1, x2, and bias -1
batch_num = 225;
epoch = 5000;
lr = 0.01;                   % learning rate
alpha = 0.9;

% input initialization 
input = trainset(:, 1:2, :);
input(:, 3, :)= -1;           % 225 x 3 
dk = trainset(:, 3, :);       % groudn truth 

test = trainset(:, 1:2, :);
test(:, 3, :)= -1;  
tk = testset(:, 3, :);  

% weights declariation : row x column x batch   
wj = normrnd(0, 1, [input_num, node_num, batch_num]);    
wk = normrnd(0, 1, [node_num, 1, batch_num]);        

x = 1;
y_nop = -inf;
y_mmt = -inf;
test_e = -inf;
train_e= -inf;

% train : Neuron j & k ,
% no memetum 
for i= 1:epoch
    % forward
    vj = pagemtimes(input , wj);           % 225 x sample_num x node_num 
    yj = sigmoid(vj);

    vk = pagemtimes(yj , wk);              
    yk = sigmoid(vk);

    error = dk - yk;         
    error_batch = pagemtimes(error, error);
    error_sum = sum(error_batch, 3)/225;

    % neuron k
    delta_k = pagemtimes(error, pagemtimes(yk, 1-yk));  
    wght_del_k = lr * pagemtimes(delta_k, yj);  

    % neuron j
    delta_j_sig = pagemtimes(wk, delta_k); 
    delta_j = diagonal(pagemtimes(yj, 'transpose', (1-yj), 'none'));
    delta_j = delta_j + delta_j_sig;
    wght_del_j = lr * pagemtimes(delta_j, input); 

    % update weight 
    wj = wj + pagetranspose(wght_del_j);
    wk = wk + pagetranspose(wght_del_k);

    % draw chart error_sum x epoch
    y_nop = [y_nop error_sum];

    test_error = cal_error(wj, wk, 1);  % test 
    train_error = cal_error(wj, wk, 0); % train

    test_e = [test_e test_error];
    train_e =[train_e train_error];

    x = [x i];
end

% compare train and test
f1 = figure('Name', 'train vesus test');
plot(x,train_e, x, test_e)

% weights declariation : row x column x batch   
wj = normrnd(0, 1, [input_num, node_num, batch_num]);    
wk = normrnd(0, 1, [node_num, 1, batch_num]);        

y_mmt = -inf;

% with mometum 
for i= 1:epoch
    % forward
    vj = pagemtimes(input , wj);           % 225 x sample_num x node_num 
    yj = sigmoid(vj);

    vk = pagemtimes(yj , wk);              
    yk = sigmoid(vk);

    error = dk - yk;         
    error_batch = pagemtimes(error, error);
    error_sum = sum(error_batch, 3)/225;

    % neuron k
    delta_k = pagemtimes(error, pagemtimes(yk, 1-yk));  
    if i >1
      wght_del_k =  lr * pagemtimes(delta_k, yj) + alpha* wght_del_k;    % momentum 
    else
      wght_del_k = lr * pagemtimes(delta_k, yj);  
    end

    % neuron j
    delta_j_sig = pagemtimes(wk, delta_k); 
    delta_j = diagonal(pagemtimes(yj, 'transpose', (1-yj), 'none'));
    delta_j = delta_j + delta_j_sig;
    
    if i >1 
        wght_del_j =  lr * pagemtimes(delta_j, input) + alpha* wght_del_j; % momentum 
    else
        wght_del_j = lr * pagemtimes(delta_j, input); 
    end

    % update weight 
    wj = wj + pagetranspose(wght_del_j);
    wk = wk + pagetranspose(wght_del_k);

    % draw chart error_sum x epoch
    y_mmt = [y_mmt, error_sum];
end
f2= figure('Name', 'momentum');
plot(x,y_nop, x, y_mmt)

function error_sum = cal_error(wj, wk, mode)
trainset = load_data('./regressor_trn.txt');
testset = load_data('./regressor_tst.txt');

input = trainset(:, 1:2, :);
input(:, 3, :)= -1;           % 225 x 3 
dk = trainset(:, 3, :);       % groudn truth 

test = trainset(:, 1:2, :);
test(:, 3, :)= -1;  
tk = testset(:, 3, :);     

    if mode==1
        vj = pagemtimes(test , wj);           % 225 x sample_num x node_num 
        yj = sigmoid(vj);

        vk = pagemtimes(yj , wk);              
        yk = sigmoid(vk);

        error = tk - yk;         
        error_batch = pagemtimes(error, error);
        error_sum = sum(error_batch, 3)/225;

    else
        vj = pagemtimes(input , wj);           % 225 x sample_num x node_num 
        yj = sigmoid(vj);

        vk = pagemtimes(yj , wk);              
        yk = sigmoid(vk);

        error = dk - yk;         
        error_batch = pagemtimes(error, error);
        error_sum = sum(error_batch, 3)/225;
    end 

end

function out = diagonal(input)    
    for i=1:225
        out(:,:,i)=  diag(input(:,:,i));
    end
end

function data = sigmoid_prime(output)
    data = output.* (1 -output);
end 

function mat = load_data(path)
    fid = fopen(path);
    for i=1:4, buffer = fgetl(fid); end % remove headers

    data = textscan(fid, '%f%f%f', 'Delimiter','\t');
    
    mat= zeros(1, 3, 225);
    mat(1, 1, :)= data{1,1};
    mat(1, 2, :)= data{1,2};
    mat(1, 3, :)= data{1,3};
    
    fclose(fid);
end

