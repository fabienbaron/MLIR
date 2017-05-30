function y = loggausspdf2(X, sigma)
[R,p]= chol(sigma);
if p ~= 0
    error('Covariance matrix not definite positive');
end
y= -sum((R'\X).^2,1)/2-size(X,1)/2*log(2*pi)-sum(log(diag(R)))/2;
