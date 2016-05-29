clear

%% load data
fid = fopen('iris.txt', 'rt');
C = textscan(fid, '%f, %f, %f, %f, %s');
fclose(fid);

data = cell2mat(C(:, 1:4));
clear C fid

%% preprocessing

data = data - repmat(mean(data), size(data,1), 1);
data = data ./ repmat(std(data), size(data,1), 1);

hold on
scatter(data(1:50,2), data(1:50,3), [], 'black')
scatter(data(51:100,2), data(51:100,3), [], 'blue')
scatter(data(101:150,2), data(101:150,3), [], 'red')
hold off

%% set parameters

maxIter = 100;
sigma_0 = .3; % the standard deviation for likelihood
sigma_1 = 3; % the standard deviation for the base measure
alpha = 3;


%% initial setting

data = data(:,2:3);
ix = ones(size(data,1), 1);
centers = zeros(size(data));

%% DP sampling
for iter = 1:maxIter
    for i = 1:size(data,1)
        ptr = ix;
        ptr(i) = [];
        tb = histcounts(ptr); % frequencies for each cluster
        
        prob = zeros(length(tb)+1, 1);
        for j = 1:length(tb)
            prob(j) = tb(j) * mvnpdf(data(i,:), centers(j,:), sigma_0^2 * eye(size(data,2)));
        end
        prob(length(tb)+1) = alpha / (2*pi) * (sigma_0^(-2) + sigma_1^(-2)) / sigma_0 / sigma_1 ...
            * exp(-1/2 * sigma_0^(-2) * sigma_1^(-2) / (sigma_0^(-2) + sigma_1^(-2)) * (data(i,:)* data(i,:)'));
        
        prob = prob / sum(prob);
        
        [~,~,ix(i)] = histcounts(rand(1), [0; cumsum(prob)]);
    end
    
    B = accumarray(ix, 1:length(ix), [], @(x){x});
    for i = 1:length(B)
        if ~isempty(B{i})
            centers(i,:) = mean(data(B{i},:));
        end
    end
    
    fprintf(['iter ', num2str(iter), ' done\n'])
end