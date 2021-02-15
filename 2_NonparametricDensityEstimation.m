% Pattern Classification
% Ch4. Nonparametric Techniques

% Data import
% Assume that x1 and x2 are samples of two classes c1 and c2
load 2_data.mat

% Grid
xx = -6:0.05:6;

% Prior probability
prior1 = 0.5;
prior2 = 1 - prior1;

%=========================================================================%
% Simple approach : Histogram of x1 and x2
%=========================================================================%
[n1, y1] = hist(x1, 50);
[n2, y2] = hist(x2, 50);

% Draw histogram
figure
hold on
hist1 = bar(y1, n1/25, 'hist');
hist2 = bar(y2, n2/25, 'hist');
set(hist1, 'FaceColor', 'b');
set(hist2, 'FaceColor', 'c');
hold off
ylim([0 1])
legend('Class 1', 'Class 2', 'Location', 'NorthWest')

%=========================================================================%
% 4.3 Parzen Windows
%=========================================================================%

% Length of an edge of hypercube
h = 0.2;

% Case of Gaussian
%=========================================================================%

% Parzen window estimates for class 1
pg1 = zeros(1, length(xx));
for k = 1:length(xx)
    for l = 1:length(x1)
        pg1(k) = pg1(k) + (1/h) * (1/sqrt(2*pi)) * exp(-((xx(k) - x1(l))/h)^2/2);
    end
    pg1(k) = (1/length(x1)) * pg1(k);
end

% Parzen window estimates for class 2
pg2 = zeros(1, length(xx));
for k = 1:length(xx)
    for l = 1:length(x2)
        pg2(k) = pg2(k) + (1/h) * (1/sqrt(2*pi)) * exp(-((xx(k) - x2(l))/h)^2/2);
    end
    pg2(k) = (1/length(x2)) * pg2(k);
end

% Plot the results
%--------------------------------------------------------------------------

% Plot conditional probability: p(x|c_i)
figure
subplot(1, 2, 1);
hold on
plot(xx, pg1, 'k');
plot(xx, pg2, ':k');
hold off
ylim([0 1])
legend('Class 1', 'Class 2', 'Location', 'NorthWest')

% Plot posterior probability: p(c_i|x)
allx_g = pg1 * prior1 + pg2 * prior2;
post1_g = (pg1 * prior1) ./ allx_g;
post2_g = (pg2 * prior2) ./ allx_g;
subplot(1, 2, 2)
hold on
plot(xx, post1_g, 'k');
plot(xx, post2_g, ':k');
hold off
ylim([0 1])
legend('Class 1', 'Class 2', 'Location', 'NorthWest')

% Case of box function
%=========================================================================%

% Parzen window estimates for class 1
pb1 = zeros(1, length(xx));
for k = 1:length(xx)
    for l = 1:length(x1)
            pb1(k) = pb1(k) + (1/h) * (abs(xx(k) - x1(l))/h <= 0.5);
    end
    pb1(k) = (1/length(x1)) * pb1(k);
end

% Parzen window estimates for class 2
pb2 = zeros(1, length(xx));
for k = 1:length(xx)
    for l = 1:length(x2)
            pb2(k) = pb2(k) + (1/h) * (abs(xx(k) - x2(l))/h <= 0.5);
    end
    pb2(k) = (1/length(x2)) * pb2(k);
end

% Plot the results
%--------------------------------------------------------------------------

% Plot conditional probability: p(x|c_i)
figure
subplot(1, 2, 1);
hold on
plot(xx, pb1, 'k');
plot(xx, pb2, ':k');
hold off
ylim([0 1])
legend('Class 1', 'Class 2', 'Location', 'NorthWest')

% Plot posterior probability: p(c_i|x)
allx_b = pb1 * prior1 + pb2 * prior2;
post1_b = (pb1 * prior1) ./ allx_b;
post2_b = (pb2 * prior2) ./ allx_b;
subplot(1, 2, 2);
hold on
plot(xx, post1_b, 'k');
plot(xx, post2_b, ':k');
hold off
ylim([0 1])
legend('Class 1', 'Class 2', 'Location', 'NorthWest')

%=========================================================================%
% 4.4 k-Nearest Neighbor Estimation
%=========================================================================%

% Number of samples in hypercube
k = 15;

% kNN estimates for class 1
for i = 1:length(xx)
    for j = 1:length(x1) 
        dist(j) = abs(xx(i) - x1(j));
    end
    temp = sort(dist);
    r = temp(k);
    pkn1(i) = k / (2 * r * length(x1)); 
end

% kNN estimates for class 2
for i = 1:length(xx)
    for j = 1:length(x2) 
        dist(j) = abs(xx(i) - x2(j));
    end
    temp = sort(dist);
    r = temp(k);
    pkn2(i) = k / (2 * r * length(x2)); 
end

% Plot the results
%--------------------------------------------------------------------------

% Plot conditional probability: p(x|c_i)
figure
subplot(1, 2, 1);
hold on
plot(xx, pkn1, 'k');
plot(xx, pkn2, ':k');
hold off
ylim([0 1])
legend('Class 1', 'Class 2', 'Location', 'NorthWest')

% Plot posterior probability: p(c_i|x)
allx_kn = pkn1 * prior1 + pkn2 * prior2;
post1_kn = (pkn1 * prior1) ./ allx_kn;
post2_kn = (pkn2 * prior2) ./ allx_kn;
subplot(1, 2, 2);
hold on
plot(xx, post1_kn, 'k');
plot(xx, post2_kn, ':k');
hold off
ylim([0 1])
legend('Class 1', 'Class 2', 'Location', 'NorthWest')
