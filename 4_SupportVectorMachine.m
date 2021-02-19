% Pattern Classification
% Ch.5.11. Support Vector Machines

%=========================================================================%
% Generate data
%=========================================================================%

% Random seed
rng(1)

% Dimension and number of samples 
d = 2; n = 100;

% Generate data set
% choice = 1 : linearly separable data
% choice = 2 : linearly non-separable data
% choice = 3 : skewed linearly separable data
% choice = 4 : quasi-linear data
choice = 1;
switch choice
    case 1 % Linearly separable
        x = 2 * rand(d, n) - [1; 1] * ones(1, n);
        l = 2 * ((2 * x(1, :) + x(2, :)) > 0.5) - 1;
    case 2 % Linearly non-separable
        x = 2 * rand(d, n) - [1; 1] * ones(1, n);
        l = 2 * xor((2 * x(1, :) + x(2, :)) > 0.5, (x(1, :) - 1.5 * x(2, :)) > 0.5) - 1;
    case 3 % Skewed linearly separable
        x = [randn(d, n/2)/4 + [0.5; 0.5] * ones(1, n/2) randn(d, n/2)/4 - [0.5; 0.5] * ones(1, n/2)];
        l = 2 * (x(1, :) > 0) - 1;
    case 4 % Quasi-linear
        x = 2 * rand(d, n) - [1; 1] * ones(1, n);
        l = 2 * ((2 * x(1, :) + x(2, :)) > 0.5) - 1;
        flip = abs((2 * x(1, :) + x(2, :)) - 0.5) < 0.2;
        l(find(flip)) = -l(find(flip));
end

%=========================================================================%
% Linear SVM
%=========================================================================%

h = x; h(:, l<0) = -h(:, l<0);

% Option
options = optimset('Algorithm','interior-point-convex');

% Quadratic Programming Optimization
alpha = quadprog(h' * h, -ones(1, size(x, 2)), [], [], l, 0, zeros(1, size(x, 2)), [], [], options)';

% Calculate w
w = sum(x .* (ones(size(x, 1), 1) * (l .* alpha)), 2);

% Support vector
sv = alpha > 1e-5;
isv = find(sv);

% Calculate b
b = sum(w' * x(:, isv) - l(isv)) / sum(sv); 

% Plot the classification result
figure
hold on
plot(x(1, find(l > 0 & sv)), x(2, find(l > 0 & sv)), 'ro', 'MarkerFaceColor', [1 0 0]);
plot(x(1, find(l > 0 & ~sv)), x(2, find(l > 0 & ~sv)), 'bo');
plot(x(1, find(l < 0 & sv)), x(2, find(l < 0 & sv)), 'r^', 'MarkerFaceColor', [1 0 0]);
plot(x(1, find(l < 0 & ~sv)), x(2, find(l < 0 & ~sv)), 'b^');
ref0 = refline(-w(1) / w(2), b / w(2));
ref1 = refline(-w(1) / w(2), (b+1) / w(2));
ref2 = refline(-w(1) / w(2), (b-1) / w(2));
set(ref0, 'Color', 'k');
set(ref1, 'Color', 'r', 'LineStyle', ':');
set(ref2, 'Color', 'r', 'LineStyle', ':');
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);
title('Linear SVM')

%=========================================================================%
% Soft Margin
%=========================================================================%

h = x; h(:, l<0) = -h(:, l<0);

c = 10 * ones(1, size(x, 2));

% Option
options = optimset('Algorithm','interior-point-convex');

% Quadratic Programming Optimization
alpha = quadprog(h' * h, -ones(1, size(x, 2)), [], [], l, 0, zeros(1, size(x, 2)), c, [], options)';

% Calculate w
w = sum(x .* (ones(size(x, 1), 1) * (l .* alpha)), 2);

% Support vector
svL = alpha > 1e-5;
svU = alpha < c(1) - (1e-5);
sv = (svL + svU) == 2;
isv = find(sv);

% Calculate b
b = sum(w' * x(:, isv) - l(isv)) / sum(sv);

% Plot the classification result
figure
hold on
plot(x(1, find(l > 0 & sv)), x(2, find(l > 0 & sv)), 'ro', 'MarkerFaceColor', [1 0 0]);
plot(x(1, find(l > 0 & ~sv)), x(2, find(l > 0 & ~sv)), 'bo');
plot(x(1, find(l < 0 & sv)), x(2, find(l < 0 & sv)), 'r^', 'MarkerFaceColor', [1 0 0]);
plot(x(1, find(l < 0 & ~sv)), x(2, find(l < 0 & ~sv)), 'b^');
ref0 = refline(-w(1) / w(2), b / w(2));
ref1 = refline(-w(1) / w(2), (b+1) / w(2));
ref2 = refline(-w(1) / w(2), (b-1) / w(2));
set(ref0, 'Color', 'k');
set(ref1, 'Color', 'r', 'LineStyle', ':');
set(ref2, 'Color', 'r', 'LineStyle', ':');
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);
title(['Soft Margin, C = ', num2str(c(1))])

%=========================================================================%
% Kernel Extension
%=========================================================================%

h = x; 

c = 10 * ones(1, size(x, 2));

% RBF Kernel
diff = zeros(size(x, 2), size(x, 2));
for i = 1 : size(x, 2)
    for j = 1 : size(x, 2)
        diff(i, j) = norm( h(:, i) - h(:, j) );
    end
end
sigma = 1;
kernel = exp( -( diff / (2 * sigma^2) ) );
y  = l' * l;
kernel = kernel .* y;

% Option
options = optimset('Algorithm','interior-point-convex');

% Quadratic Programming Optimization
alpha = quadprog(kernel, -ones(1, size(x, 2)), [], [], l, 0, zeros(1, size(x, 2)), c, [], options)'; 

% Support vector
svL = alpha > 1e-5;
svU = alpha < c(1) - (1e-5);
sv = (svL + svU) == 2;
isv = find(sv);

% Calculate b
b = (1/sum(sv)) * sum( sum( (alpha' * ones(1, size(isv, 2))) .* (l' * ones(1, size(isv, 2))) .* (kernel(isv, :)' .* y(isv, :)') ) - l(isv) );

% Grid
xx = -1.5:0.01:1.5; yy = -1.5:0.01:1.5;

% Coordinates for both sides of decision boundary
db1 = []; counter1 = 0;
db2 = []; counter2 = 0;

% Extract the coordinates for both sides of decision boundary
for i = 1:length(xx)
    for j = 1:length(yy)
        grid = [xx(i); yy(j)];
        
        % Discriminant function
        AlphaYKernel = 0;
        for k = 1:length(h)
            AlphaYKernel = AlphaYKernel + alpha(k) * l(k) * exp( -( norm(h(:, k) - grid) / (2 * sigma^2) ) );
        end
        f = AlphaYKernel - b;
        
        if f > 0
            counter1 = counter1 + 1;
            db1(counter1, :) = [xx(i), yy(j)];
        end
        if f < 0
            counter2 = counter2 + 1;
            db2(counter2, :) = [xx(i), yy(j)];
        end
        
    end
end

% plot the classification result
%--------------------------------------------------%
% CAUTION: 
% Depending on the data, the decision boundary cannot be drawn well...

figure 
hold on

% Plot decision boundary
plot(db1(:, 1), db1(:, 2), 'g'); 
plot(db2(:, 1), db2(:, 2), 'y'); 

% Plot support vectors
plot(x(1, find(l > 0 & sv)), x(2, find(l > 0 & sv)), 'ro', 'MarkerFaceColor', [1 0 0]);
plot(x(1, find(l < 0 & sv)), x(2, find(l < 0 & sv)), 'r^', 'MarkerFaceColor', [1 0 0]);

% Plot data other than support vectors
plot(x(1, find(l > 0 & ~sv)), x(2, find(l > 0 & ~sv)), 'ko', 'MarkerFaceColor', [0 0 0]);
plot(x(1, find(l < 0 & ~sv)), x(2, find(l < 0 & ~sv)), 'k^', 'MarkerFaceColor', [0 0 0]);
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);
title(['Kernel Extension, C = ', num2str(c(1)), ', sigma = ', num2str(sigma)])
