% Pattern Classification
% Ch.6. Multilayer Neural Network

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
% Multilayer Neural Network
%=========================================================================%
% Consider three-layer neural network: Input, intermediate and output layers

% Convert labels into 0 or 1 from -1 or 1
for l = 1:length(label0)
    if label0(l) == -1
        label0(l) = 0;
    end
end

% Number of Intermediate units and parameters
NumInt = 30;
NumParam = (4 * NumInt) + 1;

% Initialization
rho = 0.8; criterion = 0.01; J = 1;
weight = randn(1, NumParam);
counter = 0;

% Train the network using backpropagation
while J > criterion
    
    J = 0;
    counter = counter + 1;
        
    for i = 1 : length(aug_x)

        %====================================================%
        % Forward phase
        %====================================================%

        % From input to intermedeate layer
        %------------------------------------------%

        % Output from Input layer
        aug_g_in = aug_x(:, i)';

        % Input to Intermediate layer
        for j = 1 : NumInt
            h_int(j) = weight(:, j*(1:3)) * aug_g_in';
        end

        % Output from Intermediate layer
        g_int = 1 ./ (1 + exp(-h_int));        
        aug_g_int = [1, g_int];

        % From intermedeate to output layer
        %------------------------------------------%

        % Input to Output layer
        h_out = weight(:, (3 * NumInt + 1) : length(weight)) * aug_g_int';

        % Output from Output layer
        g_out = 1 / (1 + exp(-h_out));        
        
        %====================================================%
        % Criterion function
        %====================================================%
        Jp = (1/2) * (g_out - label0(i))^2;
        J = J + Jp;
        
        %====================================================%
        % Backward phase
        %====================================================%

        % Error at output layer
        error_out = (g_out - label0(i)) * g_out * (1 - g_out);
        
        % Update weight from intermediate to output layer
        weight(:, (3 * NumInt + 1) : length(weight)) = weight(:, (3 * NumInt + 1) : length(weight)) - rho * error_out * aug_g_int;
        
        % Error at intermediate layer
        error_int = (weight(:, (3 * NumInt + 1) : length(weight)) * error_out) .* aug_g_int .* (1 - aug_g_int);
        
        % Update weight from input to intermediate layer
        for k = 1 : NumInt
            weight(:, k*(1:3)) = weight(:, k*(1:3)) - rho * error_int(k+1) * aug_g_in;
        end
    end
    
    % Display the value of criterion function
    disp(J);
    
end

%=========================================================================%
% Display classification results
%=========================================================================%
% CAUTION: 
% Depending on the data, the decision boundary cannot be drawn well...

% Grid
xx = -1.5 : 0.01 : 1.5; yy = -1.5 : 0.01 : 1.5;

% Coordinates for both sides of decision boundary
db1 = []; counter1 = 0;
db2 = []; counter2 = 0;

% Extract the coordinates for both sides of decision boundary
for i = 1 : length(xx)
    for j = 1 : length(yy)
        
        % Input (augmented) vector
        pattern = [1; xx(i); yy(j)];
        
        % Input to Intermediate layer
        for k = 1 : NumInt
            h_int(k) = weight(:, k*(1:3)) * pattern;
        end

        % Output from Intermediate layer
        g_int = 1 ./ (1 + exp(-h_int));        
        aug_g_int = [1, g_int];

        % Input to Output layer
        h_out = weight(:, (3 * NumInt + 1) : length(weight)) * aug_g_int';

        % Output from Output layer
        g_out = 1 / (1 + exp(-h_out)); 
        
        % Extract coordinates (xx,yy) for g_out < 0.5 and 0.5 < g_out
        if g_out < 0.5
            counter1 = counter1 + 1;
            db1(counter1, :) = [xx(i), yy(j)];
        elseif 0.5 < g_out 
            counter2 = counter2 + 1;
            db2(counter2, :) = [xx(i), yy(j)];
        end
        
        %if (0.5 - 1e-3 < g_out) && (g_out < 0.5 + 1e-3)
        %    counter2 = counter2 + 1;
        %    db(counter2, :) = [xx(i), yy(j)];
        %end
    end
end

% Plot the data and decision boundary
figure
hold on
plot(db1(:, 1), db1(:, 2), 'g');                                % Plot one side of decision boundary
plot(db2(:, 1), db2(:, 2), 'y');                                % Plot other side of decision boundary
plot(x(1, find(label0 == 0)), x(2, find(label0 == 0)), 'bo');   % Plot the data for class 1
plot(x(1, find(label0 == 1)), x(2, find(label0 == 1)), 'b^');   % Plot the data for class 2
xlim([-1.5, 1.5])
ylim([-1.5, 1.5])
switch choice
    case 1 
        title(['Neural Network, Linearly separable, # of intermediate units = ', num2str(NumInt), ', iteration = ', num2str(counter)])
    case 2 
        title(['Neural Network, Linearly non-separable, # of intermediate units = ', num2str(NumInt), ', iteration = ', num2str(counter)])
    case 3 
        title(['Neural Network, Skewed linearly separable, # of intermediate units = ', num2str(NumInt), ', iteration = ', num2str(counter)])
end
hold off

