function [P1, P] = CLSR(Xs,Xt,Xs_label,alpha,beta,lambda,mu2)

Y = Construct_Y(Xs_label,length(Xs_label));
B = Construct_B(Y);
Class = length(unique(Xs_label));
Max_iter = 100;
[m,n1] = size(Xs); n2 = size(Xt,2);
max_mu = 10^7;
mu = 0.1;
rho = 1.01;
convergence = 10^-6;
options = [];
options.ReducedDim = Class;

Cs = cov(Xs') + eye(size(Xs',2));
Ct = cov(Xt') + eye(size(Xt',2));
Pc = Cs^(-1/2) * Ct^(1/2);

[P,~] = eigs(Pc, Class);

% ----------------------------------------------
%               Initialization
% ----------------------------------------------
M = ones(Class,n1);
E = zeros(Class,n2);

Z = zeros(n1,n2);
Z1 = zeros(n1,n2);
Z2 = zeros(n1,n2);

Y1 = zeros(Class,n2);
Y2 = zeros(n1,n2);
Y3 = zeros(n1,n2);

% ------------------------------------------------
%                   Main Loop
% ------------------------------------------------
for iter = 1:Max_iter
    
    
    % updating P1
    V0 = Y+B.*M;
    V00 = P'*Xt-E+Y1/mu;
    
    if (iter == 1)
        P1 = Pc;
    else
        A1 = 2*Cs*Pc*Pc'*Cs'+2*lambda*P*P'+mu2*eye(m);
        B1 = Xs*Z*Z'*Xs';  
        P1 = dlyap(-A1\(mu*P*P'),B1,A1\(2*Cs*Pc*Ct'+mu*P*V00*Z'*Xs'+mu2*Pc));
    end
    
    
    % updating Pc
    if (iter == 1)
        Pc = Pc;
    else
        Ac = 2*Cs'*P1*P1'*Cs+mu2*eye(m);
        Bc = Xs*Xs';
        Pc = dlyap(-Ac\(2*P*P'),Bc,Ac\(2*P*V0*Xs'+2*Cs'*P1*Ct+mu2*P1));
    end
    
    
    % updating P
    V1 = Xt-P1*Xs*Z;
    V2 = E-Y1/mu;
    if (iter == 1)
        P = P;
    else
        P = (2*Pc*Xs*Xs'*Pc'+lambda*eye(m)+2*lambda*P1*P1'+mu*V1*V1')\(2*Pc*Xs*V0'+mu*V1*V2');
    end
    
    
    % updating M
    R = P'*Pc*Xs-Y;
    gp = B.*R;
    [numm1,numm2] = size(gp);
    for jk1 = 1:numm1
        for jk2 = 1:numm2
            M(jk1,jk2) = max(gp(jk1,jk2),0);
        end
    end
    
    % updating E
    the2 = beta/mu;
    temp_E = P'*Xt-P'*P1*Xs*Z+Y1/mu;
    E = max(0,temp_E-the2)+min(0,temp_E+the2);
    
    % updating Z
    V3 = -Z1+Y2/mu;
    V4 = -Z2+Y3/mu;
    V5 = P'*Xt-E+Y1/mu;
    Z = (mu*Xs'*P1'*P*P'*P1*Xs+2*mu*eye(n1))\(mu*Xs'*P1'*P*V5-mu*V3-mu*V4);
    
    % updating  Z1
    ta = 1/mu;
    temp_Z1 = Z+Y2/mu;
    [U01,S01,V01] = svd(temp_Z1,'econ');
    S01 = diag(S01);
    svp = length(find(S01>ta));
    if svp >= 1
        S01 = S01(1:svp)-ta;
    else
        svp = 1;
        S01 = 0;
    end
    Z1 = U01(:,1:svp)*diag(S01)*V01(:,1:svp)';
    
    % updating Z2
    taa = alpha/mu;
    temp_Z2 = Z+Y3/mu;
    Z2 = max(0,temp_Z2-taa)+min(0,temp_Z2+taa);
    
    % updating Y1, Y2, Y3
    Y1 = Y1+mu*(P'*Xt-P'*P1*Xs*Z-E);
    Y2 = Y2+mu*(Z-Z1);
    Y3 = Y3+mu*(Z-Z2);
    
    % updating mu
    mu = min(rho*mu,max_mu);
    
    % checking convergence
    leq1 = norm(P'*Xt-P'*P1*Xs*Z-E,Inf);
    leq2 = norm(Z-Z1,Inf);
    leq3 = norm(Z-Z2,Inf);
    if iter > 2
        if leq1<convergence && leq2<convergence && leq3<convergence
            break
        end
    end
end

end

function B = Construct_B(Y)
%%
B = Y;
B(Y==0) = -1;
end

function Y = Construct_Y(gnd,num_l)
%%
nClass = length(unique(gnd));
Y = zeros(nClass,length(gnd));
for i = 1:num_l
    for j = 1:nClass
        if j == gnd(i)
            Y(j,i) = 1;
        end
    end
end
end