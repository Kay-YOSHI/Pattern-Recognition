% Pattern Classification
% 2.6 Discriminant Functions for the Normal Density

%=========================================================================%
% CASE1: ‡”_i = ƒÐ^2 * I
%=========================================================================%

% Generate Data
%--------------------------------------------------------------------------

% Random seed
rng(1);

% Dimension and sample size
d = 2; n = 1000; 

% Mean and variance-covariance matrix
m1 = [0; 3]; m2 = [3; 0];
sigma = 1; cov = sigma^2 * eye(2);

% Random sample from 2D normal distributions with mean m1 and m2
x1 = cov * randn(n, d)' + m1 * ones(1,n);
x2 = cov * randn(n, d)' + m2 * ones(1,n);

% Prior probability
p1 = 0.5; p2 = 1-p1;

% Discriminant function
%--------------------------------------------------------------------------
w = m1 - m2;
x0 = (1/2) * (m1 + m2) - ( (sigma^2) / (norm(m1 - m2)^2) * log(p1 / p2) * (m1 - m2) );
l1 = w' * (x1 - x0 * ones(1, n)) > 0; % Label for x1
l2 = w' * (x2 - x0 * ones(1, n)) > 0; % Label for x2

% Plot the classification results
%--------------------------------------------------------------------------
figure
hold on
xlim([-4, 8])
ylim([-4, 8])

% Plot the discriminant function
x11 = -4 : 0.1 : 8;
x22 = (1/w(2)) * ((-1) * w(1) * x11 + w' * x0);
plot(x11, x22, 'k');

% Contour plot
[xx, yy] = meshgrid(-4 : 0.5 : 8, -4 : 0.5 : 8);
tmp = [xx(:) yy(:)]' - m1 * ones(1, length(xx(:)));
p = 1 / (2 * pi * sqrt(det(cov))) * exp((-1/2) * diag(tmp' * inv(cov) * tmp));
contour(xx, yy, reshape(p, size(xx)), 10, 'k');
tmp = [xx(:) yy(:)]' - m2 * ones(1, length(xx(:)));
p = 1 / (2 * pi * sqrt(det(cov))) * exp((-1/2) * diag(tmp' * inv(cov) * tmp));
contour(xx, yy, reshape(p, size(xx)), 10, 'k');

% Scatter plot
plot(x1(1, find(l1)), x1(2, find(l1)), 'co');                                % correct x1
plot(x1(1, find(1-l1)), x1(2, find(1-l1)), 'ro', 'MarkerFaceColor',[1 0 0]); % incorrect x1
plot(x2(1, find(1-l2)), x2(2, find(1-l2)), 'g^');                            % correct x2
plot(x2(1, find(l2)), x2(2, find(l2)), 'r^', 'MarkerFaceColor',[1 0 0]);     % incorrect x2
hold off

% Compute the fraction of miss-classification
%--------------------------------------------------------------------------
incorrect1 = n - sum(l1);
incorrect2 = sum(l2);
disp(['CASE1: Fraction of miss-classification(%) = ' num2str(((incorrect1+incorrect2)/n)*100)])

%=========================================================================%
% CASE2: ‡”_i = ‡”
%=========================================================================%

% Generate Data
%--------------------------------------------------------------------------
rng(2); d = 2; n = 1000;
m1 = [0; 3]; m2 = [3; 0];
cov = [1 0; 0 2];
r = [cos(pi/4) -sin(pi/4); sin(pi/4) cos(pi/4)];
x1 = (randn(n, d) * cov * r)' + m1 * ones(1, n);
x2 = (randn(n, d) * cov * r)' + m2 * ones(1, n);
p1 = 0.5; p2 = 1-p1;

% Discriminant function
%--------------------------------------------------------------------------
w = m1 - m2;
x0 = (1/2) * (m1 + m2) - ( 1 / ((m1 - m2)' * inv(r' * cov * r) * (m1 - m2)) * log(p1/p2) * (m1 - m2) );
l1 = w' * (x1 - x0 * ones(1, n)) > 0; % Label for x1
l2 = w' * (x2 - x0 * ones(1, n)) > 0; % Label for x2

% Plot the classification results
%--------------------------------------------------------------------------
figure
hold on
xlim([-6, 8])
ylim([-6, 8])

% Plot the discriminant function
x11 = -10 : 0.1 : 10;
x22 = (1/w(2)) * ((-1) * w(1) * x11 + w' * x0);
plot(x11, x22, 'k');

% Contour plot
[xx, yy] = meshgrid(-10 : 0.5 : 10, -10 : 0.5 : 10);
tmp = [xx(:) yy(:)]' - m1 * ones(1, length(xx(:)));
p = 1 / (2 * pi * sqrt(det(cov))) * exp((-1/2) * diag(tmp' * inv(r' * cov * r) * tmp));
contour(xx, yy, reshape(p, size(xx)), 10, 'k');
tmp = [xx(:) yy(:)]' - m2 * ones(1, length(xx(:)));
p = 1 / (2 * pi * sqrt(det(cov))) * exp((-1/2) * diag(tmp' * inv(r' * cov * r) * tmp));
contour(xx, yy, reshape(p, size(xx)), 10, 'k');

% Scatter plot
plot(x1(1, find(l1)), x1(2, find(l1)), 'co');                                % correct x1
plot(x1(1, find(1-l1)), x1(2, find(1-l1)), 'ro', 'MarkerFaceColor',[1 0 0]); % incorrect x1
plot(x2(1, find(1-l2)), x2(2, find(1-l2)), 'g^');                            % correct x2
plot(x2(1, find(l2)), x2(2, find(l2)), 'r^', 'MarkerFaceColor',[1 0 0]);     % incorrect x2
hold off

% Compute the fraction of miss-classification
%--------------------------------------------------------------------------
incorrect1 = n - sum(l1);
incorrect2 = sum(l2);
disp(['CASE2: Fraction of miss-classification(%) = ' num2str(((incorrect1+incorrect2)/n)*100)])

%=========================================================================%
% CASE3: ‡”_i = arbitrary
%=========================================================================%

% Generate Data
%--------------------------------------------------------------------------
rng(3); d = 2; n = 1000;
m1 = [-1; 2]; m2 = [2; -2];
cov1 = [1 0; 0 2]; cov2 = [2 0; 0 1];
r1 = [cos(pi/3) -sin(pi/3); sin(pi/3) cos(pi/3)];
r2 = [cos(pi/3) -sin(pi/3); sin(pi/3) cos(pi/3)];
x1 = (randn(n, d) * cov1 * r1)' + m1 * ones(1, n);
x2 = (randn(n, d) * cov2 * r2)' + m2 * ones(1, n);
p1 = 0.5; p2 = 1-p1;

% Discriminant function
%--------------------------------------------------------------------------
W1 = (-1/2) * inv(r1' * cov1 * r1);
W2 = (-1/2) * inv(r2' * cov2 * r2);
W3 = W1 - W2;
ws1 = inv(r1' * cov1 * r1) * m1;
ws2 = inv(r2' * cov2 * r2) * m2;
ws3 = ws1 - ws2;
w10 = m1' * W1 * m1 - (1/2) * log(det(inv(r1' * cov1 * r1))) + log(p1);
w20 = m2' * W2 * m2 - (1/2) * log(det(inv(r2' * cov2 * r2))) + log(p2);
w30 = w10 - w20;

for i = 1:n
    l1(i) = x1(:, i)' * (W1 - W2) * x1(:, i) + (ws1 - ws2)' * x1(:, i) + (w10 - w20) > 0; % Label for x1
    l2(i) = x2(:, i)' * (W1 - W2) * x2(:, i) + (ws1 - ws2)' * x2(:, i) + (w10 - w20) > 0; % Label for x2
end

% Plot the classification results
%--------------------------------------------------------------------------
figure
hold on
xlim([-8, 8])
ylim([-8, 8])
[xx, yy] = meshgrid(-10 : 0.5 : 10, -10 : 0.5 : 10);

% plot the discriminant function
f = W3(1, 1) .* xx.^2 + W3(2, 1) .* xx .* yy + W3(1, 2) .* xx .* yy + W3(2, 2) .* yy.^2 + ws3(1) .* xx + ws3(2) .* yy + w30;
contour(xx, yy, f, [0, 0], 'k');

% Contour plot
tmp = [xx(:) yy(:)]' - m1 * ones(1, length(xx(:)));
p = 1 / (2 * pi * sqrt(det(cov1))) * exp((-1/2) * diag(tmp' * inv(r1' * cov1 * r1) * tmp));
contour(xx, yy, reshape(p, size(xx)), 10, 'k');
tmp = [xx(:) yy(:)]' - m2 * ones(1, length(xx(:)));
p = 1/(2 * pi * sqrt(det(cov2))) * exp((-1/2) * diag(tmp' * inv(r2' * cov2 * r2) * tmp));
contour(xx, yy, reshape(p, size(xx)), 10, 'k');

% Scatter plot
plot(x1(1, find(l1)), x1(2, find(l1)), 'co');                                % correct x1
plot(x1(1, find(1-l1)), x1(2, find(1-l1)), 'ro', 'MarkerFaceColor',[1 0 0]); % incorrect x1
plot(x2(1, find(1-l2)), x2(2, find(1-l2)), 'g^');                            % correct x2
plot(x2(1, find(l2)), x2(2, find(l2)), 'r^', 'MarkerFaceColor',[1 0 0]);     % incorrect x2
hold off

% Compute the fraction of miss-classification
%--------------------------------------------------------------------------
incorrect1 = n - sum(l1);
incorrect2 = sum(l2);
disp(['CASE3: Fraction of miss-classification(%) = ' num2str(((incorrect1+incorrect2)/n)*100)])
