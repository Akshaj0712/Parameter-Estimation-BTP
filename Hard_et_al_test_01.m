%% This is the BATCH version
%% import Hard_et_al_test_projection_quad.project_a_with_theta_constraints.m
load('JMLR data rand 2M 002.mat');
x = xt;
y = yt;

%% Gradient OR Backpropogation algorithm

% x vec(N)
% y vec(N)

% Parameters
n = 2; % dimension of the system
T = 100;
mu = 0.0001; %from paper 
N = int32(size(x,2)/T);
% Balpha =; %how do you set this?
sigma = 1; % variance for the random sample of a 
disp("ORIGINAL")
a = sigma * randn(n,1); A = [ [zeros(n-1,1) eye(n-1)]; -flipud(a)' ]

B = zeros(n, 1); B(n) = 1

C = sigma * randn(n,1).'

D = 0

norm_clip = 100;

first = true;
second = true;

% --- NEW: batch settings ---
batch_size = 100;            % number of sequences per gradient update
if batch_size <= 0
    error('batch_size must be positive');
end

% Accumulators for batch gradients
GA_acc = zeros(n,n);        % GA is matrix shaped in your code; keep same shape
GC_acc = zeros(size(C));    % 1-by-n (since you computed GC using dy*h(i,:))
GD_acc = 0;                 % scalar

% Counters
batch_count = 0;            % how many sequences accumulated in current batch
global_step = 0;            % counts how many sequences have been processed total

last_dy = [];               % will hold dy of last processed sequence (for plotting)
global_loss = [];
tic
for j = 1:N
    global_step = global_step + 1;
    % FEED FORWARD
    h0 = zeros(n,1);
    h = [(A*h0+B*x((j-1)*T + 1)).'];
    y_est = [C*h(1,:).'+D*x((j-1)*T + 1)];
    for i = 2:T
        h(i,:) = A*h(i-1,:).' + B*x((j-1)*T + i);
        y_est(i) = C*h(i,:).' + D*x((j-1)*T + i);
    end

    % BACK PROPAGATION for this one sequence
    dh = zeros(T+1, n); % preallocate (dh(i,:) for i=1..T, dh(T+1,:)=0)
    GA = zeros(n,n);
    GC = zeros(size(C)); % 1-by-n
    GD = 0;
    T1 = T/4;
    dy = zeros(T,1);

    for i = T:-1:1
        if (i > T1)
            dy(i) = y_est(i) - y((j-1)*T + i);
        else
            dy(i) = 0;
        end

        % dh(i,:) = C*dy(i) + (A*dh(i+1,:).')'  (keep shapes consistent)
        dh(i,:) = (C * dy(i)) + (A * dh(i+1,:).').' ;
        GC = GC + 1/(T - T1) * dy(i) * h(i,:);
        if (i > 1)
            GA = GA - 1/(T - T1) * (dh(i,:).') * h(i-1,:);
        else
            GA = GA - 1/(T - T1) * (dh(i,:).') * h0.';
        end
        GD = GD + 1/(T - T1) * dy(i) * x((j-1)*T + i);
    end

    % accumulate gradients into batch accumulators
    GA_acc = GA_acc + GA;
    GC_acc = GC_acc + GC;
    GD_acc = GD_acc + GD;

    batch_count = batch_count + 1;
    last_dy = dy; % store last sequence residual (for plotting after loop)
    global_loss(end+1) = norm(dy);
    % if batch full (or last sequence overall), apply update + projection
    if (batch_count == batch_size) || (j == N)
        % GRADIENT CLIPPING on the accumulated gradients
        Ga = GA_acc(n,:).'; % same extraction as before (n-th row -> a gradient)
        na = norm(Ga, 2);
        nC = norm(GC_acc, 2);
        nD = norm(GD_acc, 2);
        gnorm = sqrt(na^2 + nC^2 + nD^2); % Combined L2 norm
        if gnorm > norm_clip
            scale = norm_clip / gnorm;
            Ga = Ga * scale;
            GC_acc = GC_acc * scale;
            GD_acc = GD_acc * scale;
        else
            % If not clipped, Ga must reflect the unscaled accumulated GA_acc
            Ga = GA_acc(n,:).';
        end

        % GRADIENT STEP: note we scale learning step by 1/batch_size to keep magnitude stable
        a = A(n,:).'; % extract a from A
        a = a - mu * (Ga / batch_count);
        C = C - mu * (GC_acc / batch_count);
        D = D - mu * (GD_acc / batch_count);

        % PROJECTION (apply to new a)
        [a_proj, info] = Hard_et_al_test_projection_quad(a, 0.5, 2, 1, 0.5, 10, 240);
        A = [ [zeros(n-1,1) eye(n-1)]; -flipud(a_proj)' ]; % reinstall a into A

        % Reset batch accumulators and counter
        GA_acc = zeros(n,n);
        GC_acc = zeros(size(C));
        GD_acc = 0;
        batch_count = 0;

        % Update learning rate schedule based on progress (use global_step)
        eps_tol = 1e-3;
        v1 = 5*log(double(N))/6;
        v2 = 11*log(double(N))/12;

        if(abs(log(double(global_step)) - v1) < eps_tol && first)
            first = false;
            mu = mu / 10;
        end
        if(abs(log(double(global_step)) - v2) < eps_tol && second)
            second = false;
            mu = mu / 10;
        end
    end
end
toc

disp("END")
A
C
D

% Plot the dy from the last processed sequence
figure;
plot(global_loss, 'LineWidth', 1.5);
xlabel('Index');
ylabel('dy');
title('Plot of dy (batch-wise last sequence)');
grid on;
