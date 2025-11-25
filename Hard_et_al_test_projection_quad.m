function [a_proj, info] = Hard_et_al_test_projection_quad(a, alpha, n, tau0, tau1, tau2, numTheta)
% Projects real vector a (n x 1) to a_proj that satisfies constraints
% for z = alpha * exp(i*theta) for theta sampled on numTheta points.
%
% Inputs:
%   a        - (n x 1) real vector to project
%   alpha    - positive scalar
%   n        - integer length of a (and length of polynomial vector)
%   tau0,tau1,tau2 - scalars as in constraints
%   numTheta - number of theta sample points (recommend 360 or more)
%
% Outputs:
%   a_proj   - projected vector (n x 1)
%   info     - struct with fields: feasible (bool), QPexitflag, usedTheta, slackUsed

% if nargin < 7
%     numTheta = 720; % default dense sampling
% end

% 1) build theta grid and complex v vectors
theta = linspace(0, 2*pi, numTheta);
m = numTheta;

% Preallocate real and imag parts of v^T
ReV = zeros(m, n);
ImV = zeros(m, n);

for k = 1:m
    z = alpha * exp(1i * theta(k));
    % v = [z^(n-1); z^(n-2); ... ; z]
    exps = (-n:1:-1)';          % exponents (n-1 down to 1)
    v = z .^ exps;              % n-1 entries; matches a size (n x 1) assumed
    % note: if you want include constant term (z^0) or different ordering adjust exps
    ReV(k,:) = real(v).';
    ImV(k,:) = imag(v).';
end

% 2) form linear inequality matrices Aineq * x <= bineq
% Constraint (1): (1+tau0)*ImV(k,:)*a + ReV(k,:)*a <= 1 - eps
eps_tol = 1e-9;
A1 = ( (1+tau0) * ImV + ReV);   % m x n
b1 = ones(m,1) * (1 - eps_tol);

% Constraint (2a): ReV(k,:)*a <= 1 - tau1
A2 = ReV;
b2 = ones(m,1) * (1 - tau1);

% Constraint (2b): -ReV(k,:)*a <= tau2 - 1  (i.e. ReV(k,:)*a >= 1-tau2)
A3 = -ReV;
b3 = ones(m,1) * (tau2 - 1);

Aineq = [A1; A2; A3];
bineq = [b1; b2; b3];

% 3) Quadratic objective minimize ||x - a||^2 = 0.5*x'Hx + f'x
nVar = n;
H = 2 * eye(nVar);          % quadprog uses 1/2 x'Hx -> H=2I gives ||x-a||^2
f = -2 * a;

% 4) solve with quadprog (requires Optimization Toolbox)
options = optimoptions('quadprog','Display','none','TolFun',1e-12,'TolX',1e-12);
try
    [xsol, fval, exitflag] = quadprog(H, f, Aineq, bineq, [], [], [], [], [], options);
catch ME
    error('quadprog failed or not available: %s', ME.message);
end

info.QPexitflag = exitflag;
info.usedTheta = theta;
info.feasible = (exitflag > 0);

if info.feasible
    a_proj = xsol;
    info.slackUsed = false;
    return;
end

% If infeasible / solver failed, solve a slack-augmented problem:
% minimize ||x-a||^2 + lambda*sum(s)  s.t. Aineq*x <= bineq + s, s >= 0
% implement by augmenting variables: y = [x; s], H_aug, f_aug, Aineq_aug, bounds
lambda = 1e4;   % penalty on slack (tune if needed)
nSlack = size(Aineq,1);

H_aug = blkdiag(H, 1e-8*eye(nSlack));      % small regularization on slack
f_aug = [-2*a; lambda*ones(nSlack,1)];

% Aineq * x - I * s <= bineq  ==> [Aineq, -I] * [x;s] <= bineq
Aineq_aug = [Aineq, -eye(nSlack)];
bineq_aug = bineq;

% s >= 0  -> lower bounds
lb = [-inf(nVar,1); zeros(nSlack,1)];
ub = [];

% solve augmented QP
try
    [ysol, fval2, exitflag2] = quadprog(H_aug, f_aug, Aineq_aug, bineq_aug, [], [], lb, ub, [], options);
catch ME
    error('quadprog (slack) failed: %s', ME.message);
end

info.QPexitflag_slack = exitflag2;
info.slackUsed = true;
a_proj = ysol(1:nVar);
info.slack = ysol(nVar+1:end);
info.slack_sum = sum(info.slack);

end
% [a_proj, info] = Hard_et_al_test_projection_quad([+0.024; -0.26; +0.9], 0.5, 3, 1, 0.5, 10, 240)


% ALGORITHM tested, checck project logs "Appendix >> Datasets >> Projection"