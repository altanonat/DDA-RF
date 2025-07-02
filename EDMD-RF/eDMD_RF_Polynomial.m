clc; clear; close all;
%% Set random seed for reproducibility
rng(2025, 'twister');
%% Parameters
alpha = 0.14;
gamma = 0.10;
dt    = 0.01;
T     = 50;
t     = 0:dt:T;
% The degree of the monomial can be set below
degree = 10;
%% Step 0: A stable, non-chaotic initial point (outside chaotic basin)
x_stable = [0.5; -2.15; 0.2]; % This leads to decaying or periodic motion
perturb_scale = 0.1;          % Small local perturbation to stay non-chaotic
num_IC = 100;

% All Initian Conditions
initial_conditions = x_stable + perturb_scale * (rand(3, num_IC) - 0.5);
%% Step 1: Simulate all trajectories
N = length(t);
X_full = zeros(3, N, num_IC);
ode_options = odeset('RelTol',1e-10,'AbsTol',1e-10); % <--- RENAMED VARIABLE
for i = 1:num_IC
    [~, X] = ode45(@(t,x) RF_system(t,x,alpha,gamma), t, initial_conditions(:,i),ode_options);
    X_full(:,:,i) = X';
end
%% Step 2: Create snapshot matrices
X1 = []; X2 = [];
for i = 1:num_IC
    Xi = X_full(:,:,i);
    X1 = [X1, Xi(:,1:end-1)];
    X2 = [X2, Xi(:,2:end)];
end
%% Step 3: Based on the candidate of library functions
Phi_X1 = lift_polynomial_3var(X1, degree);
Phi_X2 = lift_polynomial_3var(X2, degree);

%% Step 4: Compute Koopman operator (approximated)
K = Phi_X2 / Phi_X1;

%% Step 5: Predict for first 3 initial conditions
exps = generate_exponents(degree);
idx_x = find(all(exps == [1,0,0], 2));
idx_y = find(all(exps == [0,1,0], 2));
idx_z = find(all(exps == [0,0,1], 2));

all_X_true = cell(1, 3);
all_X_pred = cell(1, 3);
all_rmse   = zeros(1, 3);

% Find the predicted waveforms and the RMSE
for i = 1:3
    X_true = X_full(:,:,i);
    phi = lift_polynomial_3var(X_true(:,1), degree);
    X_pred = zeros(3, N);
    for k = 1:N
        X_pred(:,k) = [phi(idx_x); phi(idx_y); phi(idx_z)];
        if k < N
            phi = K * phi;
        end
    end
    all_X_true{i} = X_true;
    all_X_pred{i} = X_pred;
    all_rmse(i) = sqrt(mean(sum((X_true - X_pred).^2, 1)));
end

%% Step 6: Plotting
% Set default figure properties for export
set(groot, 'defaultFigureUnits', 'centimeters');
set(groot, 'defaultFigurePosition', [0 0 8 6]); % [left bottom width height], adjust height as needed
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 8);
set(groot, 'defaultLineLineWidth', 1); % Set default line width to 1 for all plots

% Modified titles to reflect x1, x2, x3
titles = {'$x_1(t)$', '$x_2(t)$', '$x_3(t)$'}; % Using LaTeX for subscripts
% If you prefer plain text without LaTeX interpretation:
% titles = {'x_1(t)', 'x_2(t)', 'x_3(t)'};

for i = 1:3
    % Figure 1: 3D Trajectory, Phase Portrait, and RMSE
    f1 = figure;
    set(f1, 'Units', 'centimeters', 'Position', [0 0 8 8]); % Set specific size for this figure

    subplot(2,2,1)
    plot3(all_X_true{i}(1,:), all_X_true{i}(2,:), all_X_true{i}(3,:), 'k', 'LineWidth', 1)
    hold on
    plot3(all_X_pred{i}(1,:), all_X_pred{i}(2,:), all_X_pred{i}(3,:), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1)
    legend('True','eDMD Predicted', 'Location', 'best')
    title(sprintf('3D Trajectory (IC %d)', i))
    % Modified labels for x1, x2, x3
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex'); zlabel('$x_3$','Interpreter','latex'); grid on; view(3)
    % If using plain text titles, use:
    % xlabel('x_1'); ylabel('x_2'); zlabel('x_3'); grid on; view(3)


    subplot(2,2,2)
    plot(all_X_true{i}(1,:), all_X_true{i}(2,:), 'k', 'LineWidth', 1)
    hold on
    plot(all_X_pred{i}(1,:), all_X_pred{i}(2,:), '--', 'Color', [0.5 0.5 0.5],'LineWidth', 1)
    % Modified labels for x1, x2
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
    % If using plain text titles, use:
    % xlabel('x_1'); ylabel('x_2');
    title(sprintf('Phase Portrait $x_1$-$x_2$ (IC %d)', i), 'Interpreter', 'latex')
    legend('True','eDMD Predicted', 'Location', 'best'); grid on

    % Modified RMSE title to reflect x_1, x_2, x_3 if desired, otherwise no change needed
    subplot(2,2,3)
    bar(all_rmse(i))
    title(sprintf('RMSE (IC %d): %.4f', i, all_rmse(i)))
    ylabel('RMSE'); grid on

    % Save Figure 1
    filename1 = sprintf('figure_IC%d_overview.pdf', i);
    exportgraphics(f1, filename1, 'ContentType', 'vector');

    % Figure 2: Time Series
    f2 = figure;
    set(f2, 'Units', 'centimeters', 'Position', [0 0 8 10]); % Set specific size for this figure (adjust height as needed for 3 subplots)

    for j = 1:3
        subplot(3,1,j)
        plot(t, all_X_true{i}(j,:), 'k', 'LineWidth', 1)
        hold on
        plot(t, all_X_pred{i}(j,:), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1)
        % Modified ylabel to use the updated titles
        ylabel(titles{j}, 'Interpreter', 'latex'); legend('True', 'eDMD Predicted', 'Location', 'best'); grid on
        % If using plain text titles, use:
        % ylabel(titles{j}); legend('True', 'eDMD Predicted', 'Location', 'best'); grid on
    end
    xlabel('Time')
    sgtitle(sprintf('Time Series: True vs eDMD (IC %d)', i))

    % Save Figure 2
    filename2 = sprintf('figure_IC%d_timeseries.pdf', i);
    exportgraphics(f2, filename2, 'ContentType', 'vector');

    % Figure 3: Error Over Time
    f3 = figure;
    set(f3, 'Units', 'centimeters', 'Position', [0 0 8 5]); % Set specific size for this figure (adjust height as needed)

    error_over_time = vecnorm(all_X_pred{i} - all_X_true{i});
    plot(t, error_over_time, 'k', 'LineWidth', 1)
    xlabel('Time'); ylabel('Prediction Error (L2 norm)')
    title(sprintf('Error Over Time (IC %d)', i)); grid on

    % Save Figure 3
    filename3 = sprintf('figure_IC%d_error_over_time.pdf', i);
    exportgraphics(f3, filename3, 'ContentType', 'vector');
end
%% Print polynomial basis
syms x y z
fprintf('\nPolynomial Observables (degree ≤ %d):\n', degree)
for obs_idx = 1:size(exps,1)
    i = exps(obs_idx,1);
    j = exps(obs_idx,2);
    k = exps(obs_idx,3);
    monomial = x^i * y^j * z^k;
    fprintf('%2d: %s\n', obs_idx, char(monomial));
end

%% --- Functions ---
function dxdt = RF_system(~, x, alpha, gamma)
    dxdt = zeros(3,1);
    dxdt(1) = x(2)*(x(3) - 1 + x(1)^2) + gamma*x(1);
    dxdt(2) = x(1)*(3*x(3) + 1 - x(1)^2) + gamma*x(2);
    dxdt(3) = -2*x(3)*(alpha + x(1)*x(2));
end

function Phi = lift_polynomial_3var(x, degree)
    exps = generate_exponents(degree);
    num_terms = size(exps,1);
    N = size(x,2);
    Phi = zeros(num_terms, N);
    for k = 1:num_terms
        i = exps(k,1); j = exps(k,2); l = exps(k,3);
        Phi(k,:) = (x(1,:).^i) .* (x(2,:).^j) .* (x(3,:).^l);
    end
end

function exps = generate_exponents(degree)
    exps = [];
    for d = 0:degree
        for i = 0:d
            for j = 0:(d-i)
                k = d - i - j;
                exps = [exps; i, j, k];
            end
        end
    end
end
