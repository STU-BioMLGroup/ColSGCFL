function [X] = GraphFiltering(X,W,index,m)
%GRAPHFILTERING 

Xsub = X(:,index)';


Wsub = W(index,index);
%Wsub = abs(W(index,index));
n = size(Wsub,1);
    
D = diag(sum(Wsub));
L = eye(n)-D^(-1/2) * Wsub * D^(-1/2);

X_bar = Xsub;
for i = 1:m
    X_bar=(eye(n)-L/2)*X_bar;
end

X(:,index) = X_bar';



end

