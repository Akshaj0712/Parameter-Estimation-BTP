function [r, eigA, r_limit, l2norm, converges] = compute_r_and_norms(A, B, C, N)
% Inputs:
%   A (2x2), B (2x1), C (1x2)  (or C as 1x2 row vector)
%   N = number of terms to return (integer)
% Outputs:
%   r        - N-by-1 vector [r0; r1; ...; r_{N-1}] where r_t = C*A^t*B
%   eigA     - eigenvalues of A
%   r_limit  - pointwise limit of r_t as t->inf (0 if stable, [] otherwise)
%   l2norm   - sqrt(sum_{t=0..inf} r_t^2) if convergent, Inf otherwise
%   converges- logical indicating whether the ℓ2 sum converges

% Preallocate
r = zeros(N,1);

% compute sequence r_t = C * A^t * B for t = 0..N-1
At = eye(size(A));        % A^0
for t = 0:(N-1)
    r(t+1) = C * (At * B);
    At = A * At;          % update to A^(t+1)
end

% eigenvalues check
eigA = eig(A);

% pointwise limit r_t as t->inf:
if all(abs(eigA) < 1)
    % A^t -> 0, so r_t -> 0
    r_limit = 0;
else
    r_limit = [];   % no finite pointwise limit in general
end

% ℓ2 norm of the infinite sequence: sqrt( C * X * C' ), X solves X - A X A' = B B'
if all(abs(eigA) < 1)
    % discrete Lyapunov has unique solution
    X = dlyap(A, B*B');            % requires Control System Toolbox (or use custom solver)
    l2norm = sqrt( C * X * C' );   % scalar
    converges = true;
else
    l2norm = Inf;
    converges = false;
end
end

A = [0 1; 0.1 0];
B = [0; 1];
C = [0.6 0.8];
D = 0;
N = 1000

[r, eigA, r_limit, l2norm, converges] = compute_r_and_norms(A, B, C, N)

figure;
plot(r, 'LineWidth', 1.5);
xlabel('t');
ylabel('r_t');
title('Sequence r_t = C A^t B');
grid on;
