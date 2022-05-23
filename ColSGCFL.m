function [predictLabel, A, Q, L, U] = ColSGCFL(data,index,L,V,Q,Para)

maxIter = Para.maxIter;
numView = Para.numView;
nCluster = Para.nCluster;
lambda = Para.lambda;
m = Para.m;
k = Para.k;
alpha = Para.alpha;

%% Feature Representaion Initialization 
for i = 1:numView
    [U{i}] = GraphFilteringAR(data{i},L,index{i},alpha);
    %[U{i}] = GraphFiltering(data{i},Q{i},index{i},m);
end

for iter = 1:maxIter
    
    
    %% Similarity matrix generation for each view
    for i = 1 :numView
        Q{i} = SimilarityGeneration(U{i}, k, 0);
    end
    
    for i = 1 :numView
        [A{i}] = SimilarityGraphCompletion(V,Q{i},index{i},lambda);
    end
    
    %% Consensus Learning
    [L,V,predictLabel] = FusionSum(A, nCluster, numView);
    
    %% Refine dataset via mask graph filtering
    for i = 1 :numView
        [U{i}] = GraphFilteringAR(data{i},L,index{i},alpha);
        %[U{i}] = GraphFiltering(data{i},L,index{i},m);
    end
    
    fprintf('\n ############# iteration %d ################# \n', iter);
    
end

 fprintf('\n ############## Completed ################## \n');

end

