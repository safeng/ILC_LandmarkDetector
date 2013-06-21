function [ theta, beta, dev ] = logistic( x, y, w, ridge, option )
% LOGISTIC  Logistic Regression. 
%
% - x     : real matrix, one sample per row
% - y     : logical vector, class label, { 0, 1 }
% - w     : real vector, weighting factor
% - ridge : ridge term
% - option: struct( 'constant', 'on', 'epsilon', 1e-6, 'maxiter', 100 )
% - theta : E(Y) = 1 ./ ( 1 + exp( theta - beta * X ) )
% - beta  :
% - dev   : devariance, minus twice the log-likelihood.

% Revised by Yangzhou Du, in Jan 2013.

% http://www.cs.cmu.edu/~ggordon/IRLS-example/logistic.m

% nargin
if ~islogical( y ),  y = ( y > 0 );  end
if ~isa( x, 'double' ),  x = double( x );  end
if ~exist( 'w', 'var' ) || isempty( w ),  w = ones(size(x,1),1);  end
if ~exist( 'ridge', 'var' ) || isempty( ridge ),  ridge = 1e-3;  end
if ~exist( 'option', 'var' ),  option = [];  end
% option
if ~isfield( option, 'constant'),  option.constant = true;  end
if ~isfield( option, 'epsilon' ),  option.epsilon  = 1e-6;  end
if ~isfield( option, 'maxiter' ),  option.maxiter  = 100;   end
% homogeneous term
if option.constant,  x(:,end+1) = 1;  end
% ridge matrix
[ N, M ] = size( x );
switch numel(ridge)
    case 1,     ridgemat = speye( M ) * ridge;
    case M,     ridgemat = spdiags( ridge(:), 0, M, M );
    otherwise,  error( 'RIDGE should be length 1 or %d.', M );
end
% iteration
alpha = zeros( M, 1 );
oldexpy = -ones( size(y) );
for k = 1:option.maxiter
    adjy = x * alpha;
    expy = 1 ./ ( 1 + exp(-adjy) );
    deriv = expy .* ( 1 - expy ); 
    wadjy = w .* ( deriv .* adjy + (y-expy) );
    weights = spdiags( deriv .* w, 0, N, N );
    alpha = ( x' * weights * x + ridgemat ) \ ( x' * wadjy );
    if mean(abs(expy-oldexpy)) < option.epsilon,  break;  end
    oldexpy = expy;
end
% warning
if k == option.maxiter,
    warning( 'logistic:notconverged', 'Failed to converge' );
end
% beta & theta
if option.constant
    beta = alpha(1:end-1);  theta = -alpha(end);
else
    beta = alpha;  theta = 0;
end
% deviance
p = 1 ./ ( 1 + exp( - x * alpha ) );
dev = -2 * sum( y .* log( p ) + ( 1 - y ) .* log( 1 - p ) );
return;
