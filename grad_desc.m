function [U_D,U_T1,U_T2, nll_his] = grad_desc(X_obv,Y_obv,dims, ranks,s2,eta,total_ite)
%% initialize params
t = num2cell(dims);
[D, T1, T2] = deal(t{:});

t = num2cell(ranks);
[rD, rT1, rT2] = deal(t{:});

U_T2 = rand(T2,rT2);
U_T1 = rand(T1,rT1);
U_D = rand(D,rD);
L = length(Y_obv)/T2;

X = X_obv;
Y = Y_obv;
%% gradient descent
nll_his = 1000;

for ite = 1:total_ite
%%%%%% likelihood
U = kron(U_T2,kron(U_T1,U_D));
Ut = X*U;

K = Ut'*Ut + s2*eye(size(Ut,2));
iK = inv(K);

sv = svd(Ut);
ld = sum(log([sv.*sv;zeros(L*T2-length(sv),1)] + s2));
% K2 = Ut*Ut' + s2*eye(L*T2);
% ld = 2*sum(log(diag(chol(K2))));

nll = 0.5/s2*Y'*Y - 0.5/s2*Y'*Ut*iK*Ut'*Y + 0.5*ld + 0.5*reshape(U_T2,[],1)'*reshape(U_T2,[],1) + ...
    0.5*reshape(U_T1,[],1)'*reshape(U_T1,[],1) + 0.5*reshape(U_D,[],1)'*reshape(U_D,[],1);

if nll > -10^8 && nll < 10^8
% disp(ite)
% disp(nll);
%%%%%% gradient
% dL/dUt matrix
beta = iK*Ut'*Y;
L_Ut = Ut*(iK+beta*beta'/s2) - Y*beta'/s2;

L_Utt_X = L_Ut' * X;

% dUt/dU_T2ij  matrix
L_U_T2 = zeros(T2,rT2);
for i = 1:T2
    for j = 1:rT2
        L_U_T2(i,j) = trace(L_Utt_X * kron(sparse(i,j,1,T2,rT2),kron(U_T1,U_D)));
    end
end
L_U_T2_l2 = L_U_T2 + U_T2;

% dUt/dU_T1ij  matrix
L_U_T1 = zeros(T1,rT1);
for i = 1:T1
    for j = 1:rT1
        L_U_T1(i,j) = trace(L_Utt_X * kron(U_T2,kron(sparse(i,j,1,T1,rT1),U_D)));
    end
end
L_U_T1_l2 = L_U_T1 + U_T1;

% dUt/dU_Dij   matrix
L_U_D = zeros(D,rD);
for i = 1:D
    for j = 1:rD
        L_U_D(i,j) = trace(L_Utt_X * kron(kron(U_T2,U_T1),sparse(i,j,1,D,rD)));
    end
end
L_U_D_l2 = L_U_D + U_D;

%%%%%% record
if mod(ite,10)==0
    fprintf('iter %d %d\n', ite, nll);
    nll_his = [nll_his;nll];
end

%%%%%% update
step_T2 = eta/sqrt(ite);
step_T1 = eta/sqrt(ite);
step_D = eta/sqrt(ite);

U_T2 = U_T2 .* (1 - L_U_T2_l2 * step_T2);
U_T1 = U_T1 .* (1 - L_U_T1_l2 * step_T1);
U_D = U_D .* (1 - L_U_D_l2 * step_D);


end
end


end

