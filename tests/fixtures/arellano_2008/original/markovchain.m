function S = markovchain(P,T,S0);
%markovchain  generates a realization of a Markov chain with transition probability matrix Pi.
%
% S = markovchain(Pi,T) returns a T x 1 vector S, which is a realization of a Markov chain 
%     with transition probability matrix Pi.
%
% S = markovchain(Pi,T,S0) uses the initial state S0 for S(1).
%          
% Outputs: S  = A sequence of length T of numbers between 1 and n, where n is the number of shocks
%               or the dimension of Pi
% Inputs:  Pi = Transition probability matrix
%          T  = length of time series
%          S0 = Initial state for S(1), where S0 is in the set 1,2,...,n.
%
% Torben Mark Pedersen
% March 1998
% UNDERSTANDING BUSINESS CYCLES, ch. 8, Copenhagen 1998
%

[n,nc] = size(P);
S      = zeros(T,1);

if nargin < 3;
  S(1,1)  = ceil(rand*n);
else
  S(1,1)  = S0;
end;

C     = cumsum(P');            % Creates a matrix with cumulated sum over rows, (Pi') 

for t = 2:T;
  j      = find(rand < C(:,S(t-1)));
  S(t,1) = j(1);
end;
