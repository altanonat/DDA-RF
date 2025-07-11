clc; clear; close all;

%% Set a constant random seed for reproducibility
rng(2025, 'twister');

%% Parameters
alpha = 0.14;
gamma = 0.10;
dt    = 0.01;
T     = 50;
tspan = 0:dt:T;

% Number of initial conditions
num_initial_points = 100;

% Store results for multiple initial conditions
all_X_true    = cell(num_initial_points, 1);
all_X_pred_nn = cell(num_initial_points, 1);
all_rmse_nn   = zeros(num_initial_points, 1); % Changed from all_mse_nn
%% Step 0: A stable, non-chaotic initial point (outside chaotic basin)
x_stable = [0.5; -2.15; 0.2]; % This leads to decaying or periodic motion
perturb_scale = 0.1;          % Small local perturbation to stay non-chaotic
num_IC = 100;

initial_conditions = x_stable + perturb_scale * (rand(3, num_IC) - 0.5);
%% Step 1: Generate Data for Training (first IC)
fprintf('Generating training/validation sets...\n');
ode_options = odeset('RelTol',1e-10,'AbsTol',1e-10);
[t_single, X_single] = ode45(@(t,x) RF_system(t,x,alpha,gamma), tspan, initial_conditions(:,1), ode_options);
X_single = X_single';
% 80% is being used for the training
numTimeSteps = size(X_single, 2);
numTrain = floor(0.8 * (numTimeSteps - 1));

X_train = X_single(:, 1:numTrain);
Y_train = X_single(:, 2:numTrain+1);
X_val   = X_single(:, numTrain+1:end-1);
Y_val   = X_single(:, numTrain+2:end);

X_train = X_train';
Y_train = Y_train';
X_val   = X_val';
Y_val   = Y_val';

layers = [
    featureInputLayer(3, 'Name', 'input')
    fullyConnectedLayer(32, 'Name', 'fc1')         % 3×32 + 32 = 128
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')         % 32×64 + 64 = 2112
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(64, 'Name', 'fc3')         % 64×64 + 64 = 4160
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(3, 'Name', 'fc_out')       % 64×3 + 3 = 195
    regressionLayer('Name', 'output')];

training_options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 50);

%% Step 3: Train or Load Network
doTrain = false;
if doTrain
    fprintf('Training neural network...\n');
    net = trainNetwork(X_train, Y_train, layers, training_options);
    save('rf_nn_model.mat', 'net');
else
    fprintf('Loading pre-trained neural network...\n');
    load('rf_nn_model.mat', 'net');
end

%% Step 4: Predict and Evaluate RMSE
fprintf('Simulating and predicting for %d initial conditions...\n', num_initial_points);
for k = 1:num_initial_points
    fprintf('Processing IC %d/%d...\n', k, num_initial_points);
    x0 = initial_conditions(:, k);
    [t, X_true] = ode45(@(t,x) RF_system(t,x,alpha,gamma), tspan, x0, ode_options);
    X_true = X_true';

    X_in = X_true(:,1:end-1)';
    X_pred = predict(net, X_in)';
    X_true_clipped = X_true(:,1:end-1);

    rmse = sqrt(mean(sum((X_pred - X_true_clipped).^2, 1)));
    all_X_true{k} = X_true_clipped;
    all_X_pred_nn{k} = X_pred;
    all_rmse_nn(k) = rmse;
end
%% Plot the predicted and true time series
% Set default figure properties for export
set(groot, 'defaultFigureUnits', 'centimeters');
set(groot, 'defaultFigurePosition', [0 0 8 6]); % Initial default size, adjusted later for specific figures
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 8);
set(groot, 'defaultLineLineWidth', 1); % Set default line width to 1 for all plots

% Define state titles using LaTeX for subscripts
titles = {'$x_1(t)$', '$x_2(t)$', '$x_3(t)$'};
num_plot = min(3, num_initial_points); % Assuming num_initial_points is defined
for i = 1:num_plot
    % --- Figure 1: 3D Trajectory, Phase Portrait, and RMSE ---
    f1 = figure;
    % Adjust figure size for the 2x2 subplot layout (e.g., 8cm width, 8cm height)
    set(f1, 'Units', 'centimeters', 'Position', [0 0 8 8]);

    subplot(2,2,1)
    % True: Black solid, NN Predicted: Grey dashed
    plot3(all_X_true{i}(1,:), all_X_true{i}(2,:), all_X_true{i}(3,:), 'k', 'LineWidth', 1)
    hold on
    plot3(all_X_pred_nn{i}(1,:), all_X_pred_nn{i}(2,:), all_X_pred_nn{i}(3,:), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1)
    legend('True','NN Predicted', 'Location', 'best')
    title(sprintf('3D Trajectory (IC %d)', i))
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex'); zlabel('$x_3$','Interpreter','latex'); grid on; view(3)

    subplot(2,2,2)
    % True: Black solid, NN Predicted: Grey dashed
    plot(all_X_true{i}(1,:), all_X_true{i}(2,:), 'k', 'LineWidth', 1)
    hold on
    plot(all_X_pred_nn{i}(1,:), all_X_pred_nn{i}(2,:), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1)
    xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex');
    title(sprintf('Phase Portrait $x_1$-$x_2$ (IC %d)', i), 'Interpreter', 'latex');
    legend('True','Predicted', 'Location', 'best'); grid on % 'Predicted' vs 'NN Predicted' for brevity in legend

    subplot(2,2,3)
    bar(all_rmse_nn(i))
    title(sprintf('RMSE (IC %d): %.4f', i, all_rmse_nn(i)))
    ylabel('RMSE'); grid on

    % Save Figure 1
    filename1 = sprintf('ANN_figure_IC%d_overview.pdf', i);
    exportgraphics(f1, filename1, 'ContentType', 'vector');


    % --- Figure 2: Time Series ---
    f2 = figure;
    % Adjust figure size for the 3x1 subplot layout (e.g., 8cm width, 10cm height)
    set(f2, 'Units', 'centimeters', 'Position', [0 0 8 10]);

    for j = 1:3
        subplot(3,1,j)
        % True: Black solid, NN Predicted: Grey dashed
        plot(t(1:end-1), all_X_true{i}(j,:), 'k', 'LineWidth', 1) % LineWidth set to 1
        hold on
        plot(t(1:end-1), all_X_pred_nn{i}(j,:), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1) % LineWidth set to 1
        ylabel(titles{j}, 'Interpreter', 'latex')
        legend('True', 'ANN Predicted', 'Location', 'best'); grid on
    end
    xlabel('Time')
    sgtitle(sprintf('Time Series: True vs ANN (IC %d)', i)) % Changed sgtitle for clarity

    % Save Figure 2
    filename2 = sprintf('ANN_figure_IC%d_timeseries.pdf', i);
    exportgraphics(f2, filename2, 'ContentType', 'vector');

    % --- Figure 3: Prediction Error Over Time ---
    f3 = figure;
    set(f3, 'Units', 'centimeters', 'Position', [0 0 8 5]); % Adjust height as needed

    err = vecnorm(all_X_pred_nn{i} - all_X_true{i});
    % Error plot typically black
    plot(t(1:end-1), err, 'k', 'LineWidth', 1) % LineWidth set to 1
    xlabel('Time'); ylabel('Prediction Error (L2 norm)') % Added (L2 norm) for consistency
    title(sprintf('Error Over Time (IC %d)', i)); grid on

    % Save Figure 3
    filename3 = sprintf('ANN_figure_IC%d_error_over_time.pdf', i);
    exportgraphics(f3, filename3, 'ContentType', 'vector');
end
%% RF System
function dx = RF_system(~, x, alpha, gamma)
    dx = zeros(3,1);
    dx(1) = x(2)*(x(3)-1 + x(1)^2) + gamma*x(1);
    dx(2) = x(1)*(3*x(3)+1 - x(1)^2) + gamma*x(2);
    dx(3) = -2*x(3)*(alpha + x(1)*x(2));
end
