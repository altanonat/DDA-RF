clear all; close all; clc
%% Parameters
alpha = 0.14;
gamma = 0.10;
n     = 3; % Size of the states (outputs as well)
dt    = 0.01;
tspan = dt:dt:50;
N = length(tspan);
polyorder = 4;
lambda = 0.025;
usesine = false;  % If it is desired to add trigonometric terms to the library
%% Base IC and generate 3 slightly perturbed initial conditions
x_stable = [0.5; -2.15; 0.2]; % Stable and non-chaotic
perturb_scale = 0.1;
initial_conditions = x_stable + perturb_scale * (rand(3, 3) - 0.5); % 3 initial conditions
%% Simulate data for learning using first IC
x0_learn = x_stable;
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[t, x_temp] = ode45(@(t,x) RF_system(t,x,alpha,gamma), tspan, x0_learn, options);
x = x_temp';   % 3 x samples
%% Compute Derivatives
dx = zeros(n, length(t));
for i = 1:length(t)
    dx(:,i) = RF_system(0, x(:,i), alpha, gamma);
end
%% Build library and check sizes
Theta = poolData(x', n, polyorder, usesine);
Xi = sparsifyDynamics(Theta, dx', lambda, n);

fprintf('Size Theta: %d x %d\n', size(Theta));     % features x samples
fprintf('Size dx: %d x %d\n', size(dx));           % states x samples
fprintf('Size Theta'': %d x %d\n', size(Theta'));  % samples x features
fprintf('Size dx'': %d x %d\n', size(dx'));        % samples x states
%% Display identified equations
poolDataLIST({'x','y','z'}, Xi, n, polyorder, usesine);
%% Predict trajectories for the first 3 initial conditions
titles = {'x(t)', 'y(t)', 'z(t)'};
rmse   = zeros(n,1);
for ic = 1:3
    % Ground truth
    [~, x_true_temp] = ode45(@(t,x) RF_system(t,x,alpha,gamma), tspan, initial_conditions(:,ic), options);
    x_true = x_true_temp';  % 3 x N

    % Initialize prediction
    x_pred = zeros(n, N);
    x_pred(:,1) = initial_conditions(:,ic);

    for k = 2:N
        phi1 = poolData(x_pred(:,k-1)', n, polyorder, usesine);   % 1 x m
        dx1 = phi1 * Xi;                                 % 1 x 3

        x_temp = x_pred(:,k-1) + dt * dx1';              % Euler step
        phi2 = poolData(x_temp', n, polyorder, usesine);          % 1 x m
        dx2 = phi2 * Xi;                                 % 1 x 3

        % Heun' Method (Improved Euler Method)
        x_pred(:,k) = x_pred(:,k-1) + dt * 0.5 * (dx1' + dx2');
    end

    % Set default figure properties for export
    set(groot, 'defaultFigureUnits', 'centimeters');
    set(groot, 'defaultFigurePosition', [0 0 8 6]); % Initial default size, adjusted later for specific figures
    set(groot, 'defaultAxesFontName', 'Times New Roman');
    set(groot, 'defaultAxesFontSize', 8);
    set(groot, 'defaultLineLineWidth', 1); % Set default line width to 1 for all plots

    % Define state titles using LaTeX for subscripts
    titles = {'$x_1(t)$', '$x_2(t)$', '$x_3(t)$'};

    % Assuming 'ic' is part of a loop, e.g., for ic = 1:num_ics
    % And x_true, x_pred, tspan are defined for each 'ic'
    % And 'rmse' is a pre-allocated array (e.g., rmse = zeros(1, num_ics);)

    % --- Figure 1: 3D Trajectory, Phase Portrait, and RMSE ---
    f1 = figure;
    % Adjust figure size for the 2x2 subplot layout (e.g., 8cm width, 8cm height)
    set(f1, 'Units', 'centimeters', 'Position', [0 0 8 8]);

    subplot(2,2,1)
    % True: Black solid, SINDy Predicted: Grey dashed
    plot3(x_true(1,:), x_true(2,:), x_true(3,:), 'k', 'LineWidth', 1); hold on;
    plot3(x_pred(1,:), x_pred(2,:), x_pred(3,:), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    legend('True','SINDy Predicted', 'Location', 'best'); grid on; view(3)
    title(sprintf('3D Trajectory (IC %d)', ic));
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex'); zlabel('$x_3$','Interpreter','latex');

    subplot(2,2,2)
    % True: Black solid, SINDy Predicted: Grey dashed
    plot(x_true(1,:), x_true(2,:), 'k', 'LineWidth', 1); hold on;
    plot(x_pred(1,:), x_pred(2,:), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    title(sprintf('Phase Portrait $x_1$-$x_2$ (IC %d)', ic), 'Interpreter', 'latex');
    legend('True', 'SINDy', 'Location', 'best');
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex'); grid on;

    subplot(2,2,3)
    % Ensure rmse(ic) is calculated before plotting if it's not already
    rmse(ic) = sqrt(mean(sum((x_true - x_pred).^2, 1))); % Assuming this is inside an IC loop
    bar(rmse(ic)); % Plot only the current IC's RMSE
    title(sprintf('RMSE (IC %d): %.4f', ic, rmse(ic))); % Display current IC's RMSE value
    ylabel('RMSE'); grid on;

    % Save Figure 1
    filename1 = sprintf('SINDy_figure_IC%d_overview.pdf', ic);
    exportgraphics(f1, filename1, 'ContentType', 'vector');


    % --- Figure 2: Time Series ---
    f2 = figure;
    % Adjust figure size for the 3x1 subplot layout (e.g., 8cm width, 10cm height)
    set(f2, 'Units', 'centimeters', 'Position', [0 0 8 10]);

    for j = 1:3
        subplot(3,1,j)
        % True: Black solid, SINDy: Grey dashed
        plot(tspan, x_true(j,:), 'k', 'LineWidth', 1); hold on;
        plot(tspan, x_pred(j,:), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
        ylabel(titles{j}, 'Interpreter', 'latex'); legend('True', 'SINDy', 'Location', 'best'); grid on;
    end
    xlabel('Time')
    sgtitle(sprintf('Time Series: True vs SINDy (IC %d)', ic));

    % Save Figure 2
    filename2 = sprintf('SINDy_figure_IC%d_timeseries.pdf', ic);
    exportgraphics(f2, filename2, 'ContentType', 'vector');

    % --- Figure 3: Prediction Error Over Time ---
    f3 = figure;
    set(f3, 'Units', 'centimeters', 'Position', [0 0 8 5]); % Adjust height as needed

    % Error plot typically black
    error_over_time = vecnorm(x_pred - x_true);
    plot(tspan, error_over_time, 'k', 'LineWidth', 1)
    xlabel('Time'); ylabel('Prediction Error (L2 norm)')
    title(sprintf('Error Over Time (IC %d)', ic)); grid on

    % Save Figure 3
    filename3 = sprintf('SINDy_figure_IC%d_error_over_time.pdf', ic);
    exportgraphics(f3, filename3, 'ContentType', 'vector');
end
%% --- Dynamics Function ---
function dx = RF_system(~, x, alpha, gamma)
dx = zeros(3,1);
dx(1) = x(2)*(x(3) - 1 + x(1)^2) + gamma * x(1);
dx(2) = x(1)*(3*x(3) + 1 - x(1)^2) + gamma * x(2);
dx(3) = -2*x(3)*(alpha + x(1)*x(2));
end
