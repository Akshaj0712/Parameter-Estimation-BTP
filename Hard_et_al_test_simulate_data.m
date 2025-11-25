% Parameters
N = 2e6;      % number of time steps

n=2;
a = [0.09 0]; A = [ [zeros(n-1,1) eye(n-1)]; -flipud(a) ]
disp(size(A))
B = zeros(n, 1); B(n) = 1;
C = [0.6,0.8]
D = 1;

n = size(A,1);   % state dimension
m = size(B,2);   % input dimension
p = size(C,1);   % output dimension

% Preallocate
yt  = zeros(p, N);
xt  = zeros(m, N);

% Initial state
h = zeros(n,1);        % h0

tic
for t = 1:N
    if(mod(t,1e6)==0)
        disp(t)
    end
    % ----- Generate Gaussian noise xt -----
    x = randn(m,1);          % mean 0, variance 1

    % normalize to have l2 norm = 0.5
    x = 0.5 * x / norm(x);

    % store xt
    xt(:,t) = x;

    % ----- Compute yt = C h_t + D x_t -----
    y = C*h + D*x;
    yt(:,t) = y;

    % ----- Compute h_{t+1} = A h_t + B x_t -----
    h = A*h + B*x;

    % store h_{t+1} (will be h(t))
end
toc
% Save results (MAT format)
save('JMLR data rand 2M 002.mat','xt','yt','-v7.3'); % -v7.3 allows to save files larger than 2GB
disp('file saved')