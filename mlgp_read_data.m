function [X_obv, Y_obv] = mlgp_read_data(X_raw, Y_raw, subjectIndices,aspectIndices, dims)
%  read multi-way data into sparse matrices

t = num2cell(dims);
[D,T1, T2] = deal(t{:});

T = T1*T2;
L = length(Y_raw)/T2;
L_T = zeros(1,T1);
for t=1:T1
    L_T(t) = sum(subjectIndices==t)/3;
end


X_T = mat2cell(zeros(L,T1*D),L_T,ones(1,T1)*D);
l = ones(1,T1);
for i = 1:L
    t=subjectIndices(i*3);
    X_T{t,t}(l(t),:) = X_raw(:,i*3)';
    l(t) = l(t)+1;
end
X_obv = sparse(kron(eye(T2),cell2mat(X_T))); % note that X is kron(eye(3), mat(X_T))

Y = cell(T,1);
l = ones(T1,T2);
for i = 1:L*3
    t1 = subjectIndices(i);
    t2 = aspectIndices(i);
    Y{T1*(t2-1)+t1}(l(t1,t2),1) = Y_raw(i);
    l(t1,t2) = l(t1,t2)+1;
end
Y_obv = cell2mat(Y);
clear X_raw Y_raw X Y

end