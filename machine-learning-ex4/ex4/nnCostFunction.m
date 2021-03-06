function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(size(X,1),1) X];

z2 = X * Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2,1),1) a2];

z3 = a2 * Theta2';
hyp = sigmoid(z3);

pos_case = -y .* log(hyp);

ones_vector = ones(size(y,1),1);

total = 0;

for i=1:size(y,1)

	y_k = zeros(num_labels,1);
	y_k(y(i)) = 1;
	y_k = y_k;
	hyp_k = hyp(i,:)';
	pos_case = -y_k .* log(hyp_k);
	%size(pos_case)
	
	ones_vector = ones(num_labels, 1);
	
	neg_case= (ones_vector-y_k) .* log(ones_vector - hyp_k);
	
	total = total + sum(pos_case - neg_case);
 
end 

J = total / m;

%cost with regularization 
Theta1_no_bias = Theta1(:,(2:size(Theta1,2)));

Theta2_no_bias = Theta2(:,(2:size(Theta2,2)));

J = J + (lambda * (sum(sum(Theta1_no_bias.^2)) + sum(sum(Theta2_no_bias.^2)))) / (2*m);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

accum = 0;

%%%%%pre-processing%%%%%%
%matrix of single values 
y_matrix = eye(num_labels)(y,:);

%unbiased parameters
Theta2_unbiased = Theta2(:,2:end);
Theta1_unbiased = Theta1(:,2:end);
%%%%%%%%%%%%%%%%%%%%%%%%%

%forward propagation 
a_1_biased = X;
a_1_unbiased = a_1_biased(:, 2:end);
z_2 = a_1_biased * Theta1';
a_2_unbiased = sigmoid(z_2);

a_2_biased = [ones(size(a_2_unbiased,1),1) a_2_unbiased];
%size(a_2_biased) % 5000x26
z_3 = a_2_biased * Theta2';
%size(z_3) % 5000 x 10 

a_3 = sigmoid(z_3);
%size(a_3)
%size(y_matrix)

small_delta_3 = a_3 - y_matrix;

%size(small_delta_3) 5000x10


g_prime_z2 = sigmoidGradient(z2);
%size(g_prime_z2) %5000x25

small_delta_2 = (small_delta_3 * Theta2_unbiased) .* g_prime_z2;
%size(small_delta_2) 5000x25 

big_delta_2 = small_delta_3' * a_2_biased;
%size(small_delta_3) 5000x10 
%size(a_2_biased) 5000x26

big_delta_1 = small_delta_2' * a_1_biased;
%size(small_delta_2) 5000 x 25
%size(a_1_biased) 5000 x 401

%without regularization
Theta1_grad = big_delta_1 / m; % 25 x 401
Theta2_grad = big_delta_2 / m; % 10 x 26

%Add regularization 
regularized_1 = (lambda / m) * Theta1(:,2:end);
regularized_1 = [zeros(size(Theta1,1),1) regularized_1];

regularized_2 = (lambda / m) * Theta2(:,2:end);
regularized_2 = [zeros(size(Theta2,1),1) regularized_2];

Theta1_grad = Theta1_grad + regularized_1;
Theta2_grad = Theta2_grad + regularized_2;

%%%%%failed implementation
%for t=1:m
	
	%step_1 
%	a_1_biased = X(t,:);
%	a_1_unbiased = a_1_biased(:, 2:end);
%	%size(a_1_unbiased)
%	%a_1_biased = [1 a_1];
%	z_2 = a_1_biased * Theta1';
%	a_2 = sigmoid(z_2);
%	
%	a_2_biased = [1 a_2];
%	z_3 = a_2_biased * Theta2';
%	
%	%1 x 10 
%%	a_3 = sigmoid(z_3);
%
%	%step_2
%	y_t = [1 2 3 4 5 6 7 8 9 10] == y(t);
%	small_delta_3 = (a_3 - y_t)';
%	size(small_delta_3);
%	%step 3 
%	Theta2_no_bias = Theta2(:, 2:end);
%	
%	small_delta_2 =  (Theta2_no_bias' * small_delta_3) .* sigmoidGradient(z_2)';
%	size(small_delta_2);
%	size(a_1_unbiased);
	
	%accum = accum + small_delta_2 * a_1_unbiased
%end 

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
