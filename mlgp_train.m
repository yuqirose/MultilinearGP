function [model,L_list ] = mlgp_train( X_obv, Y_obv, dims,ranks, varargin)
%MLGP_TRAIN: training using MLTP algorithm
%output model

params = inputParser; %rD,rT1,rT2,s2,total_ite, sT2,sT1,sD

addParameter(params,'s2', 1e-2, @isscalar);
addParameter(params,'eta', 1e-2, @isscalar);
addParameter(params,'max_iter', 1e2, @isscalar);

params.parse(varargin{:});

s2= params.Results.s2;
eta= params.Results.eta;
max_iter = params.Results.max_iter;
%% gradient descent step

[U_D,U_T1,U_T2,L_list] = grad_desc(X_obv,Y_obv,dims, ranks,s2,eta,max_iter);

model.U_D = U_D;
model.U_T1 = U_T1;
model.U_T2 = U_T2;
end

