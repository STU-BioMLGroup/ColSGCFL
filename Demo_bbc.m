clear all
clc
addpath(genpath(pwd))

load bbcIncomplete.mat
numView = length(data);
nCluster = length(unique(truelabel{1}));
m = 5;  
k = 20;

%% Dataset Normalization
data = NormalizeFeature(data,numView);

%% Initialization (individual similarity matrix, unified similarity matrix and vector V)
[L,V,Q] = Initialization(data,index,nCluster,k);

Para = [];
Para.m = 5;
Para.k = 20;
Para.alpha = 5;
Para.lambda = 0.1;
Para.numView = numView;
Para.maxIter = 3;
Para.nCluster = nCluster;
[predictLabel, A, Q, L,U] = ColSGCFL(data,index,L,V,Q,Para);

FinalResult = ClusteringMeasure(truelabel{1}, predictLabel);  

fprintf('\n ###### Prediction results: ACC=%.4f, NMI=%.4f ####### \n', FinalResult(1), FinalResult(2));


ind = 1;
Representation = U{ind}(:,index{ind})';
label = truelabel{ind}(index{ind});
tsne(Representation, label, 2)



