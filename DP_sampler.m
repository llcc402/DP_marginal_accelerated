function [ix, centers] = DP_sampler(data, alpha, maxIter)

if nargin < 3
    maxIter = 100;
end

ix = ones(size(data));
centers = zeros(size(data)) + mean(data);

for iter = 1:maxIter
    for i = 1:length(data)
        
        ptr = ix;
        ptr(i) = [];
        tb = histcounts(ptr);
        
        % the prob for exit clusters
        prob = zeros(1, length(tb) + 1);
        for j = 1:length(tb)
            prob(j) = tb(j) * normpdf(data(i), centers(j), 1);
        end
        % the prob for new cluster
        prob(length(tb)+1) = alpha / 2 / pi / 3 * exp(-1/2 * data(i)^2 * (2 * 3^2 + 1) / (3^2 + 1));
        
        % normalize prob
        prob = log(prob);
        prob = prob - max(prob);
        prob = exp(prob);
        prob = prob / sum(prob);
        
        % sample index
        [~,~,ix(i)] = histcounts(rand(1), [0, cumsum(prob)]);
        
    end
    
    % update centers
    B = accumarray(ix', 1:length(ix), [], @(x){x});
    for i = 1:length(B)
        if ~isempty(B{i})
            centers(i) = mean(data(B{i}));
        end
    end
    
    fprintf(['iter ', num2str(iter), ' done\n'])
end
        
