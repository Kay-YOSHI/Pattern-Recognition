% Pattern Classification
% Ch5.5. Minimizing the Perceptron Criterion Function

%=========================================================================%
% Generate data
%=========================================================================%

% Random seed
rng(1);

% Dimension and number of samples
d = 2; n = 100; 

% Generate data set
% choice = 1 : linearly separable data
% choice = 2 : linearly non-separable data
% choice = 3 : skewed linearly separable data
choice = 1;
switch choice
    case 1 % Linearly separable
        x = 2 * rand(d, n) - [1; 1] * ones(1, n);
        label0 = 2 * ((2 * x(1, :) + x(2, :)) > 0.5) - 1;
    case 2 % Linearly non-separable
        x = 2 * rand(d, n) - [1; 1] * ones(1, n);
        label0 = 2 * xor((2 * x(1, :) + x(2, :)) > 0.5, (x(1, :) - 1.5 * x(2, :)) > 0.5) - 1;
    case 3 % Skewed linearly separable
        x = [randn(d, n/2)/4 + [0.5; 0.5] * ones(1, n/2) randn(d, n/2)/4 - [0.5; 0.5] * ones(1, n/2)];
        label0 = 2 * (x(1, :) > 0) - 1;
end

% Augmented feature/weight vectors
aug_x = [ones(1, n); x];
aug_w = rand(1, d + 1);

% Normalization
norm_x = repmat(label0, 3, 1) .* aug_x;

%=========================================================================%
% 5.5.1 Batch Perceptron
%=========================================================================%

% discriminant function
xx = -1:0.01:1; 
yy = (-1) * (aug_w(2)/aug_w(3)) * xx - (aug_w(1)/aug_w(3));

% correct or incorrect label for each class
g = aug_w * norm_x;
label1 = double(g > 0); 
label2 = label0 + label1;

% Initial plot
% Red  : incorrectly classified samples
% Blue : correctly classified samples
figure
hold on
plot(x(1, find(label2 == 2)), x(2, find(label2 == 2)), 'bo');
plot(x(1, find(label2 == 1)), x(2, find(label2 == 1)), 'ro', 'MarkerFaceColor',[1 0 0]);
plot(x(1, find(label2 == 0)), x(2, find(label2 == 0)), 'b^');
plot(x(1, find(label2 == -1)), x(2, find(label2 == -1)), 'r^', 'MarkerFaceColor',[1 0 0]);
plot(xx, yy, 'k');
xlim([-1.5, 1.5])
ylim([-1.5, 1.5])
hold off
switch choice
    case 1 
        title('Linearly separable, iteration = 0')
    case 2 
        title('Linearly non-separable, iteration = 0')
    case 3 
        title('Skewed linearly separable, iteration = 0')
end

%===============================================%
% Gradient Descent
%===============================================%

% Initialization
rho = 0.01; theta = 0.0001; iteration = 0; weight = aug_w;

% Initial ƒÏ‡”X-tilde
stepsize = rho * (sum(norm_x(:, find(label2 == 1)), 2) + sum(norm_x(:, find(label2 == -1)), 2));

% Iteration
while (abs(stepsize(1)) > theta) || (abs(stepsize(2)) > theta) || (abs(stepsize(3)) > theta)

    iteration = iteration + 1;

    % update weight
    weight = weight + stepsize';

    % update label
    g = weight * norm_x;
    label1 = double(g > 0);
    label2 = label0 + label1;

    % update discriminant function
    yy = (-1) * (weight(2)/weight(3)) * xx - (weight(1)/weight(3));

    % update plot
    figure
    hold on
    plot(x(1, find(label2 == 2)), x(2, find(label2 == 2)), 'bo');
    plot(x(1, find(label2 == 1)), x(2, find(label2 == 1)), 'ro', 'MarkerFaceColor',[1 0 0]);
    plot(x(1, find(label2 == 0)), x(2, find(label2 == 0)), 'b^');
    plot(x(1, find(label2 == -1)), x(2, find(label2 == -1)), 'r^', 'MarkerFaceColor',[1 0 0]);
    plot(xx, yy, 'r');
    xlim([-1.5, 1.5])
    ylim([-1.5, 1.5])
    hold off
    switch choice
    case 1 
        title(['Linearly separable, iteration = ', num2str(iteration)])
    case 2 
        title(['Linearly non-separable, iteration = ', num2str(iteration)])
    case 3 
        title(['Skewed linearly separable, iteration = ', num2str(iteration)])
    end

    % update ƒÏ‡”X-tilde
    stepsize = rho * (sum(norm_x(:, find(label2 == 1)), 2) + sum(norm_x(:, find(label2 == -1)), 2));
end

%=========================================================================%
% 5.5.2 Fixed-Increment Single-Sample Perceptron
%=========================================================================%

% discriminant function
xx = -1:0.01:1;
yy = (-1) * (aug_w(2)/aug_w(3)) * xx - (aug_w(1)/aug_w(3));

% correct or incorrect label for each class
g = aug_w * norm_x;
label1 = double(g > 0);
label2 = label0 + label1;

% Initial plot
figure
hold on
plot(x(1, find(label2 == 2)), x(2, find(label2 == 2)), 'bo');
plot(x(1, find(label2 == 1)), x(2, find(label2 == 1)), 'ro', 'MarkerFaceColor',[1 0 0]);
plot(x(1, find(label2 == 0)), x(2, find(label2 == 0)), 'b^');
plot(x(1, find(label2 == -1)), x(2, find(label2 == -1)), 'r^', 'MarkerFaceColor',[1 0 0]);
plot(xx, yy, 'k');
xlim([-1.5, 1.5])
ylim([-1.5, 1.5])
hold off
switch choice
    case 1 
        title('Linearly separable, iteration = 0')
    case 2 
        title('Linearly non-separable, iteration = 0')
    case 3 
        title('Skewed linearly separable, iteration = 0')
end

%===============================================%
% On-line Gradient Descent
%===============================================%

% Initialization
iteration = 0; weight = aug_w;

% Initial X-tilde
x_tilde = [norm_x(:, find(label2 == 1)), norm_x(:, find(label2 == -1))];

% Iteration
while size(x_tilde, 2) > 0

    iteration = iteration + 1;
    
    % Choose one misclassified sample
    i = mod(iteration, size(x_tilde, 2)) + 1;
    %i = randi(size(x_tilde, 2));

    % updated weight
    weight = weight + x_tilde(:, i)';

    % updated label
    g = weight * norm_x;
    label1 = double(g > 0);
    label2 = label0 + label1;

    % updated discriminant function
    yy = (-1) * (weight(2)/weight(3)) * xx - (weight(1)/weight(3));

    % update plot
    figure
    hold on
    plot(x(1, find(label2 == 2)), x(2, find(label2 == 2)), 'bo');
    plot(x(1, find(label2 == 1)), x(2, find(label2 == 1)), 'ro', 'MarkerFaceColor',[1 0 0]);
    plot(x(1, find(label2 == 0)), x(2, find(label2 == 0)), 'b^');
    plot(x(1, find(label2 == -1)), x(2, find(label2 == -1)), 'r^', 'MarkerFaceColor',[1 0 0]);
    plot(xx, yy, 'r');
    xlim([-1.5, 1.5])
    ylim([-1.5, 1.5])
    hold off
    switch choice
    case 1 
        title(['Linearly separable, iteration = ', num2str(iteration)])
    case 2 
        title(['Linearly non-separable, iteration = ', num2str(iteration)])
    case 3 
        title(['Skewed linearly separable, iteration = ', num2str(iteration)])
    end

    % updated X-tilde
    x_tilde = [norm_x(:, find(label2 == 1)), norm_x(:, find(label2 == -1))];
end

%=========================================================================%
% 5.8.1 Minimum Squared Error (MSE) Classifier
%=========================================================================%

% updated weight vector
weight_mse = inv(aug_x * aug_x') * aug_x * label0';

% discriminant function
xx = -1:0.01:1;
yy = (-1) * (weight_mse(2)/weight_mse(3)) * xx - (weight_mse(1)/weight_mse(3));

% correct or incorrect label
g = weight_mse' * norm_x;
label1 = double(g > 0);
label2 = label0 + label1;

% Plot the classification results
figure
hold on
plot(x(1, find(label2 == 2)), x(2, find(label2 == 2)), 'bo');
plot(x(1, find(label2 == 1)), x(2, find(label2 == 1)), 'ro', 'MarkerFaceColor',[1 0 0]);
plot(x(1, find(label2 == 0)), x(2, find(label2 == 0)), 'b^');
plot(x(1, find(label2 == -1)), x(2, find(label2 == -1)), 'r^', 'MarkerFaceColor',[1 0 0]);
plot(xx, yy, 'k');
xlim([-1.5, 1.5])
ylim([-1.5, 1.5])
hold off
switch choice
    case 1 
        title('MSE, Linearly separable')
    case 2 
        title('MSE, Linearly non-separable')
    case 3 
        title('MSE, Skewed linearly separable')
end

%=========================================================================%
% 5.8.4 The Widrow-Hoff or Least Mean Squared (LMS) Procedure
%=========================================================================%

% discriminant function
xx = -1:0.01:1;
yy = (-1) * (aug_w(2)/aug_w(3)) * xx - (aug_w(1)/aug_w(3));

% correct or incorrect label for each class
g = aug_w * norm_x;
label1 = double(g > 0);
label2 = label0 + label1;

% Initial plot
figure
hold on
plot(x(1, find(label2 == 2)), x(2, find(label2 == 2)), 'bo');
plot(x(1, find(label2 == 1)), x(2, find(label2 == 1)), 'ro', 'MarkerFaceColor',[1 0 0]);
plot(x(1, find(label2 == 0)), x(2, find(label2 == 0)), 'b^');
plot(x(1, find(label2 == -1)), x(2, find(label2 == -1)), 'r^', 'MarkerFaceColor',[1 0 0]);
plot(xx, yy, 'k');
xlim([-1.5, 1.5])
ylim([-1.5, 1.5])
hold off
switch choice
    case 1 
        title('Linearly separable, iteration = 0')
    case 2 
        title('Linearly non-separable, iteration = 0')
    case 3 
        title('Skewed linearly separable, iteration = 0')
end

%===============================================%
% LMS
%===============================================%
% Initialization
iteration = 0; weight = aug_w; rho = 0.05;

% Initial X-tilde
x_tilde = [aug_x(:, find(label2 == 1)), aug_x(:, find(label2 == -1))];

% Iteration
while size(x_tilde, 2) > 0
    
    % Choose a sample
    iteration = mod(iteration, size(aug_x, 2)) + 1;
    
    % updated weight
    weight = weight - rho * (weight * aug_x(:, iteration) - label0(iteration)) * aug_x(:, iteration)';
    
    % updated label
    g = weight * norm_x;
    label1 = double(g > 0);
    label2 = label0 + label1;
    
    % updated discriminant function
    yy = (-1) * (weight(2)/weight(3)) * xx - (weight(1)/weight(3));
    
    % update plot
    figure
    hold on
    plot(x(1, find(label2 == 2)), x(2, find(label2 == 2)), 'bo');
    plot(x(1, find(label2 == 1)), x(2, find(label2 == 1)), 'ro', 'MarkerFaceColor',[1 0 0]);
    plot(x(1, find(label2 == 0)), x(2, find(label2 == 0)), 'b^');
    plot(x(1, find(label2 == -1)), x(2, find(label2 == -1)), 'r^', 'MarkerFaceColor',[1 0 0]);
    plot(xx, yy, 'k');
    xlim([-1.5, 1.5])
    ylim([-1.5, 1.5])
    hold off
    switch choice
    case 1 
        title(['Linearly separable, iteration = ', num2str(iteration)])
    case 2 
        title(['Linearly non-separable, iteration = ', num2str(iteration)])
    case 3 
        title(['Skewed linearly separable, iteration = ', num2str(iteration)])
    end
    
    % updated X-tilde
    x_tilde = [norm_x(:, find(label2 == 1)), norm_x(:, find(label2 == -1))];
end

