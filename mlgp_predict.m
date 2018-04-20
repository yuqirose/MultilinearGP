function [ Y_pred, V_pred, MSE ] = mlgp_predict( X_obv, Y_obv, train_ind, test_ind, dims,model,varargin )
%MLGP_PREDICT: predictive distribution of GP
params = inputParser; %s2
addParameter(params,'s2', 1e-2, @isscalar);
params.parse(varargin{:});
s2= params.Results.s2;
%%
U_D = model.U_D;
U_T1 = model.U_T1 ;
U_T2 = model.U_T2 ;

t = num2cell(dims);
[D,T1, T2] = deal(t{:});
N = size(X_obv,1);

U = kron(U_T2,kron(U_T1,U_D)); % T1 and T2
Ut = X_obv*U;
K_clean = Ut*Ut';
K_noise = K_clean + s2*eye(N);
cov = K_noise(train_ind,train_ind);

% test
cov_test = K_clean(test_ind,train_ind);

Y_test = Y_obv(test_ind);
Y_test_h = cov_test * (cov \ Y_obv(train_ind));
ub = 0.5; lb = -0.5;
Y_test_c = (Y_test_h>ub) - (Y_test_h<lb);


Y_pred = Y_test_h;
V_pred = cov_test;
MSE = sum((Y_test_h-Y_test).^2)/length(Y_test);
end

