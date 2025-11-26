% Simulation: produce numSeqs sequences, each of length T (h0 = 0)
rng(0);               % for reproducibility (optional)

% --- User parameters ---
numSeqs = 2e5;        % number of sequences (set N here)
T = 500;              % length of each sequence
% -----------------------

% System definition (kept from your code)
n = 2;
a = [0.09 0];
A = [ [zeros(n-1,1) eye(n-1)]; a ];
B = zeros(n, 1); B(n) = 1;
C = [0.6, 0.8];
D = 1;

% dims
n = size(A,1);
m = size(B,2);
p = size(C,1);

% preallocate 3-D arrays: (dimension x time x sequence)
xt = zeros(m, T, numSeqs);
yt = zeros(p, T, numSeqs);
ht = zeros(n, T, numSeqs);   % will store h1...hT for each sequence

% simulation loop
tic;
for j = 1:numSeqs
    % initial state h0 = 0
    h = zeros(n,1);
    for t = 1:T
        % generate gaussian noise and normalize to l2-norm = 0.5
        x = randn(m,1);
        x = 0.5 * x / norm(x);

        % store input
        xt(:, t, j) = x;

        % output at time t: y_t = C*h_t + D*x_t
        y = C * h + D * x;
        yt(:, t, j) = y;

        % state update h_{t+1} = A*h_t + B*x_t
        h = A * h + B * x;

        % store h_{t+1} as h_t in ht (we store h1..hT)
        ht(:, t, j) = h;
    end

    % optional progress display every 10 sequences
    if mod(j,10) == 0
        fprintf('Completed %d / %d sequences\n', j, numSeqs);
    end
end
toc;

% Save results
outname = sprintf('JMLR_data_rand_%d_T%d.mat', numSeqs, T);
save(outname, 'xt', 'yt', '-v7.3');
fprintf('Saved %s (xt, yt)\n', outname);
