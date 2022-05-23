clear all
clc
addpath(genpath(pwd))

load bbcsportIncomplete.mat
numView = length(data);
nCluster = length(unique(truelabel{1}));
m = 3;  
k = 10;

%% Dataset Normalization
data = NormalizeFeature(data,numView);

%% Initialization (individual similarity matrix, unified similarity matrix and vector V)
[L,V,Q] = Initialization(data,index,nCluster,k);

Para = [];
Para.m = 3;
Para.k = 10;
Para.alpha = 3;
Para.lambda = 0.1;
Para.numView = numView;
Para.maxIter = 5;

Para.nCluster = nCluster;
[predictLabel, A, Q, L,U] = ColSGCFL(data,index,L,V,Q,Para);

FinalResult = ClusteringMeasure(truelabel{1}, predictLabel);  

fprintf('\n ###### Prediction results: ACC=%.4f, NMI=%.4f ####### \n', FinalResult(1), FinalResult(2));


ind = 1;
Representation = U{ind}(:,index{ind})';
label = truelabel{ind}(index{ind});
tsne(Representation, label, 2)



