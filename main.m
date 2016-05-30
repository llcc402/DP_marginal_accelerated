clear

%% load data
fid = fopen('iris.txt', 'rt');
C = textscan(fid, '%f, %f, %f, %f, %s');
fclose(fid);

data = cell2mat(C(:, 1:4));
clear C fid

%% preprocessing

[U, S, V] = svd(data);
data = U(:,1:2);
data = data ./ repmat(std(data), size(data,1), 1);

% hold on
% scatter(data(1:50,2), data(1:50,3), [], 'black')
% scatter(data(51:100,2), data(51:100,3), [], 'blue')
% scatter(data(101:150,2), data(101:150,3), [], 'red')
% hold off

%% set parameters

maxIter = 100;
lambda_0 = 1; % the standard deviation for the base measure
lambda_1 = 2.5; % the standard deviation for likelihood
alpha = 1;


%% initial setting
ix = ones(size(data,1), 1);
centers = zeros(size(data)) + repmat(mean(data), size(data,1), 1);

%% DP sampling
for iter = 1:maxIter
    for i = 1:size(data,1)
        ptr = ix;
        ptr(i) = [];
        tb = histcounts(ptr); % frequencies for each cluster
        
        prob = zeros(length(tb)+1, 1);
        for j = 1:length(tb)
            if tb(j) > 0
                prob(j) = log(tb(j)) - lambda_1^2/2 * ...
                    (data(i,:) - centers(j,:)) * (data(i,:) - centers(j,:))';
            end
        end
        prob(length(tb)+1) = log(alpha) + 2*log(lambda_0) - log(lambda_0^2 + lambda_1^2) ...
            - lambda_1^2 * lambda_0^2 / 2 / (lambda_0^2 + lambda_1^2) ...
            * data(i,:) * data(i,:)';
        
        prob = prob - max(prob);
        prob = exp(prob);
        prob = prob / sum(prob);
        
        [~,~,ix(i)] = histcounts(rand(1), [0; cumsum(prob)]);
    end
    
    B = accumarray(ix, 1:length(ix), [], @(x){x});
    for i = 1:length(B)
        if ~isempty(B{i})
            if length(B{i}) == 1
                centers(i,:) = data(B{i},:);
            else
                centers(i,:) = mean(data(B{i},:));
            end
        end
    end
    
    fprintf(['iter ', num2str(iter), ' done\n'])
end