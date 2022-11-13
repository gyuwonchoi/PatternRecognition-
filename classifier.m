clear;
clc;

% load data with normalization 
trainset = load_data('./iris97.txt');
class = load_class('./iris97.txt');
 
% divide each class to trainset and testset 
trainload= cat(3, trainset(:, :, 1:25), trainset(:, :, 51:75), trainset(:, :, 101:125));
trainload(:,5,:)=-1;

testload= cat(3, trainset(:, :, 26:50), trainset(:, :, 76:100), trainset(:, :, 126:150));
testload(:,5,:)=-1;

traincls= cat(3, class(:, :, 1:25), class(:, :, 51:75), class(:, :, 101:125));
testcls = cat(3, class(:, :, 26:50), class(:, :, 76:100), class(:, :, 126:150));

% shuffle
seed= randperm(length(trainload));
trainload=trainload(:,:,seed);
traincls=traincls(:,: ,seed);

% one-hot coding 
traincls=one_hot(traincls);
testcls=one_hot(testcls);

% training parameters
epoch = 50000;
lr = 0.01;                    
alpha = 0.9;

input_num = 5;
hidden_num = 5;
output_num = 3;
sample_num = 75;

% plot
x =1;
y_nop = inf;
test_e = inf;
train_e = inf;

% weights declariation : row x column x batch   
wj = normrnd(0, 1, [input_num, hidden_num, sample_num]);    
wk = normrnd(0, 1, [hidden_num, output_num, sample_num]);    

for i= 1:epoch
    % forward
    vj = pagemtimes(trainload , wj);           % 225 x sample_num x node_num 
    yj = sigmoid(vj);

    vk = pagemtimes(yj , wk);              
    yk = sigmoid(vk);

    error = traincls - yk;

    error_= pagetranspose(error);
    error_batch = pagemtimes(error, error_);
    error_sum = sum(error_batch, 3)/75;

    % neuron k
    delta_k = pagemtimes(error,  pagemtimes(yk, 'transpose', 1-yk, 'none')); 
    wght_del_k = lr * pagemtimes(delta_k,'transpose', yj, 'none'); 

    % neuron j
    delta_j_sig = pagemtimes(wk, 'none', delta_k, 'transpose' ); 
    delta_j = diagonal(pagemtimes(yj, 'transpose', (1-yj), 'none'));
    delta_j = delta_j + (delta_j_sig);
    wght_del_j = lr * pagemtimes(delta_j, trainload); 

    % update weight 
    wj = wj + pagetranspose(wght_del_j);
    wk = wk + pagetranspose(wght_del_k);

    % draw chart error_sum x epoch
    y_nop = [y_nop error_sum];

    test_error = cal_error(testload, testcls, wj, wk);  % test 
    train_error = cal_error(trainload, traincls, wj, wk); % train

    test_e = [test_e test_error];
    train_e =[train_e train_error];

    x = [x i];
end

f1 = figure('Name', 'train vesus test');
plot(x,train_e, x, test_e)

% with momentum 
wj = normrnd(0, 1, [input_num, hidden_num, sample_num]);    
wk = normrnd(0, 1, [hidden_num, output_num, sample_num]);    
y_mmt = inf 

for i= 1:epoch
    % forward
    vj = pagemtimes(trainload , wj);           % 225 x sample_num x node_num 
    yj = sigmoid(vj);

    vk = pagemtimes(yj , wk);              
    yk = sigmoid(vk);

    error = traincls - yk;

    error_= pagetranspose(error);
    error_batch = pagemtimes(error, error_);
    error_sum = sum(error_batch, 3)/75;

    % neuron k
    delta_k = pagemtimes(error,  pagemtimes(yk, 'transpose', 1-yk, 'none')); 
    if i >1
      wght_del_k =  lr * pagemtimes(delta_k,'transpose', yj, 'none') + alpha* wght_del_k;      % momentum 
    else
      wght_del_k = lr * pagemtimes(delta_k,'transpose', yj, 'none'); 
    end

    % neuron j
    delta_j_sig = pagemtimes(wk, 'none', delta_k, 'transpose' ); 
    delta_j = diagonal(pagemtimes(yj, 'transpose', (1-yj), 'none'));
    delta_j = delta_j + (delta_j_sig);

    if i >1
      wght_del_j =  lr * pagemtimes(delta_j, trainload) + alpha* wght_del_j;      % momentum 
    else
      wght_del_j = lr * pagemtimes(delta_j, trainload);  
    end

    % update weight 
    wj = wj + pagetranspose(wght_del_j);
    wk = wk + pagetranspose(wght_del_k);

    % draw chart error_sum x epoch
    y_mmt = [y_mmt error_sum];
end

f1 = figure('Name', 'momentum');
plot(x, y_nop, x, y_mmt)

function error_sum = cal_error(dataload, class, wj, wk)
    vj = pagemtimes(dataload , wj);            
    yj = sigmoid(vj);

    vk = pagemtimes(yj , wk);              
    yk = sigmoid(vk);

    error = class - yk;

    error_= pagetranspose(error);
    error_batch = pagemtimes(error, error_);
    error_sum = sum(error_batch, 3)/75;
end

function mat=one_hot(input)
    for i=1:75
        label = input(:,:,i);
        
        if strcmp(label,'Iris-setosa')
            mat(1,:,i)= [1 0 0];
        elseif strcmp(label,'Iris-versicolor')
            mat(1,:,i)= [0 1 0];
        elseif strcmp(label,'Iris-virginica')
            mat(1,:,i)= [0 0 1];
        end
    end
end

function mat= load_data(path)
    fid = fopen(path);
    for i=1:66, buffer = fgetl(fid); end % remove headers

    stat = [4.3 7.9 2.0 4.4 1.0 6.9 0.1 2.5]; % min max for normalization 
    data = textscan(fid, '%f%f%f%f%s', 'Delimiter',',');
   
    mat(1, 1, :)= (data{1,1}-stat(1))/(stat(2)-stat(1));
    mat(1, 2, :)= (data{1,2}-stat(3))/(stat(4)-stat(3));
    mat(1, 3, :)= (data{1,3}-stat(5))/(stat(6)-stat(5));
    mat(1, 4, :)= (data{1,4}-stat(7))/(stat(8)-stat(7));

    fclose(fid);
end


function mat= load_class(path)
    fid = fopen(path);
    for i=1:66, buffer = fgetl(fid); end % remove headers

    data = textscan(fid, '%f%f%f%f%s', 'Delimiter',',');
   
    mat(1, 1, :)= data{1,5};

    fclose(fid);
end

function out = diagonal(input)    
    for i=1:75
        out(:,:,i)=  diag(input(:,:,i));
    end
end

