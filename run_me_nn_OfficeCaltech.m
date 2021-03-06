clear all
close all

src_str = {'Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgt_str = {'amazon','webcam','dslr','Caltech10','webcam','dslr','Caltech10','amazon','dslr','Caltech10','amazon','webcam'};

load('data\Office_Caltech10_SURF\optimal_parameters_nn.mat');


for i = 1:length(tgt_str)
    src = src_str{i};
    tgt = tgt_str{i};
    fprintf(' %s vs %s ', src, tgt);
    
    load(['data\Office_Caltech10_SURF\' src '_SURF_L10.mat']);
    Xs = fts';
    Xs_label = labels;
    clear fts;
    clear labels;
    
    load(['data\Office_Caltech10_SURF\' tgt '_SURF_L10.mat']);
    Xt = fts';
    Xt_label = labels;
    clear fts;
    clear labels;
    
    % ------------------------------------------
    %             Transfer Learning
    % ------------------------------------------
    Xs = Xs./repmat(sqrt(sum(Xs.^2)),[size(Xs,1) 1]);
    Xt = Xt./repmat(sqrt(sum(Xt.^2)),[size(Xt,1) 1]);
     
    
    %                   CLSR2
    % ------------------------------------------ 
    [P1, P] = CLSR2(Xs,Xt,Xs_label,alpha(i),beta(i),lambda(i),mu2(i));
    X_train = P'*P1*Xs;
    X_test  = P'*Xt;
    
 
    % -------------------------------------------
    %               Classification
    % ------------------------------------------- 

    %                  CLSR2
    % ------------------------------------------
    X_train = X_train./repmat(sqrt(sum(X_train.^2)),[size(X_train,1) 1]);
    X_test  = X_test ./repmat(sqrt(sum(X_test.^2)),[size(X_test,1) 1]);
    mdl = fitcknn(X_train', Xs_label, 'NumNeighbors', 1);
    pred = predict(mdl, X_test');
    acc = sum(Xt_label == pred)/numel(Xt_label)*100;
    fprintf(' %2.2f%%\n',acc);
    
       
end