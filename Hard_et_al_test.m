% import Hard_et_al_test_projection_quad.project_a_with_theta_constraints.m
load('JMLR data rand 2M 001.mat');
x = xt;
y = yt;
%% Gradient OR Backpropogation algorithm

% x vec(N)
% y vec(N)

% Parameters
n = 2; % dimension of the system
T = 500;
mu = 0.0001; %from paper 
N = int32(size(x,2)/T)
% Balpha =; %how do you set this?
sigma = 1; % variance for the random sample of a 
disp("ORIGINAL")
a = sigma * randn(n,1); A = [ [zeros(n-1,1) eye(n-1)]; -flipud(a)' ]

B = zeros(n, 1); B(n) = 1;

C = sigma * randn(n,1).'

D = 0;

norm_clip = 1000;


first = true;
second = true;

for j=1:N
    % FEED FORWARD
    h0 = zeros(n,1);
    h = [(A*h0+B*x(1)).'];
    y_est = [C*h(1,:).'+D*x(1+(j-1)*T)];
    for i=2:T
        h(i,:) = A*h(i-1,:).' + B*x(i+(j-1)*T);
        y_est(i) = C*h(i,:).' + D*x(i+(j-1)*T);
    end
    
    % BACK PROPAGATION
    dh = [zeros(n,1).'];
    dh(T+1,:) = zeros(n,1).';
    GA = 0;
    GC = 0;
    GD = 0;
    T1 = T/4;
    for i=T:-1:1
        if(i>T1)
            dy(i)=y_est(i)-y(i+(j-1)*T);
        else
            dy(i) = 0;
        end
        dh(i,:) = C*dy(i)+(A*dh(i+1,:).').';
        GC = GC + 1/(T-T1)*dy(i)*h(i,:);
        if (i>1) 
            GA = GA - 1/(T-T1)*dh(i,:).'*h(i-1,:) ;
        else 
            GA = GA - 1/(T-T1)*dh(i,:).'*h0.';
        end
        GD = GD + 1/(T-T1)*dy(i)*x(i+(j-1)*T);
    end
    
    % GRADIENT CLIPPING
    Ga = GA(n,:).';
    na = norm(Ga, 2);
    nC = norm(GC, 2);
    nD = norm(GD, 2);
    gnorm = sqrt(na^2 + nC^2 + nD^2); % Combined L2 norm
    if gnorm > norm_clip
        scale = norm_clip / gnorm;
        Ga = Ga * scale;
        GC = GC * scale;
        GD = GD * scale;
    end
    
    % GRADIENT
    a = A(n,:).'; % extract a from A
    a = a - mu * Ga;
    C = C - mu*GC;
    D = D - mu*GD;
    
    % PROJECTION
    [a_proj, info] = Hard_et_al_test_projection_quad(a, 0.5, 2, 1, 0.5, 10, 240);
    A = [ [zeros(n-1,1) eye(n-1)]; -flipud(a_proj)' ]; % reinstall a into A

    eps_tol = 1e-3;

    v1 = 5*log(double(N))/6;
    v2 = 11*log(double(N))/12;
    
    if(abs(log(double(j)) - v1) < eps_tol && first)
        first = false;
        mu = mu / 10;
    end
    if(abs(log(double(j)) - v2) < eps_tol && second)
        second = false;
        mu = mu / 10;
    end

end

disp("END")
A
C
D

figure;
plot(dy, 'LineWidth', 1.5);
xlabel('Index');
ylabel('dy');
title('Plot of dy');
grid on;
