clear;clc;

load 'RestaurantDataset.mat'

D = size(X,1); %feature dimension
T1 = length(unique(subjectIndices));
T2 = length(unique(aspectIndices));
dims = [D,T1,T2];
eta = 1e-3; %learning rate
max_iter = 2e2;
ranks = [2,2,2];

[X_obv, Y_obv]  = mlgp_read_data(X,Y,subjectIndices, aspectIndices, dims);
%% train-test split
N = size(X_obv,1);
data_ind = randperm(N);
train_ratio = 0.8;
train_size = ceil(N*train_ratio);
train_ind = data_ind(1:train_size);
test_ind = data_ind(train_size+1:end);

%% training
[model, L_list ] = mlgp_train( X_obv, Y_obv, dims,ranks, 'eta', eta, 'max_iter', max_iter);

[ Y_pred, V_pred, MSE ] = mlgp_predict(X_obv, Y_obv, train_ind, test_ind, dims, model );
%% plot results
fprintf('mse %d', MSE);
plot(L_list(10:end));
xlabel('iteration');
ylabel('negative log likelihood');