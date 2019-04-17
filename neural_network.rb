require 'numo/narray'
require 'numo/gsl'
require "numo/linalg"

def matmul(a, b); Numo::Linalg.matmul(a, b); end
def exp(z); Numo::NMath.exp(z); end
def log(z); Numo::NMath.log(z); end
def sqrt(z); Numo::NMath.sqrt(z); end
def mean(m); Numo::GSL::Stats.mean(m); end

def sigmoid(z)
    1 / (1 + exp(-z))
end

def sigmoid_gradient(sigmoid)
    sigmoid * (1 - sigmoid)
end

def softmax(logits)
    exponentials = exp(logits)
    exponentials_sum = exponentials.sum(axis: 1)
    exponentials / exponentials_sum.reshape(exponentials_sum.size, 1)
end

def prepend_bias(m)
    m.insert(0, 1, axis: 1)
end

def forward(x, w1, w2)
    h = sigmoid(matmul(prepend_bias(x), w1))
    y_hat = softmax(matmul(prepend_bias(h), w2))
    [y_hat, h]
end

def back(x, y, y_hat, w2, h)
    w2_gradient = matmul(prepend_bias(h).transpose, (y_hat - y)) / x.shape[0]
    w1_gradient = matmul(
                    prepend_bias(x).transpose,
                    matmul(y_hat - y, w2[1..-1, true].transpose) * sigmoid_gradient(h)
                  ) / x.shape[0]
    [w1_gradient, w2_gradient]
end

def classify(x, w1, w2)
    y_hat, _ = forward(x, w1, w2)
    labels = y_hat.max_index(axis: 1) % y_hat.shape[1]
    labels.reshape(labels.shape[0], 1)
end

def initialize_weights(n_input_variables, n_hidden_nodes, n_classes)
    w1_rows = n_input_variables + 1
    w1 = Numo::DFloat.new(w1_rows, n_hidden_nodes).rand_norm * sqrt(1.0 / w1_rows)

    w2_rows = n_hidden_nodes + 1
    w2 = Numo::DFloat.new(w2_rows, n_classes).rand_norm * sqrt(1.0 / w2_rows)

    [w1, w2]
end

def report(iteration, x_train, y_train, x_test, y_test, w1, w2)
    classifications = classify(x_test, w1, w2)
    accuracy = (mean(classifications.eq(y_test)) * 100.0).round(2)
    puts "#{iteration} > #{accuracy}%"
end

def train(x_train, y_train, x_test, y_test, n_hidden_nodes:, iterations:, lr:)
    n_input_variables = x_train.shape[1]
    n_classes = y_train.shape[1]
    w1, w2 = initialize_weights(n_input_variables, n_hidden_nodes, n_classes)
    iterations.times do |iteration|
        report(iteration, x_train, y_train, x_test, y_test, w1, w2)
        y_hat, h = forward(x_train, w1, w2)
        w1_gradient, w2_gradient = back(x_train, y_train, y_hat, w2, h)
        w1 = w1 - (w1_gradient * lr)
        w2 = w2 - (w2_gradient * lr)
    end
    [w1, w2]
end

def one_hot_encode(labels, number_of_classes: 10)
    result = Numo::Int32.zeros(labels.size, number_of_classes)
    labels.each_with_index {|label, i| result[i, label] = 1 }
    result
end

require 'datasets'
print "Loading data..."
x_train =  Numo::NArray[*Datasets::MNIST.new(type: :train).to_a.map(&:pixels)]
y_train_raw = Datasets::MNIST.new(type: :train).to_a.map(&:label)
y_train = one_hot_encode(Numo::NArray[*y_train_raw].reshape(y_train_raw.size, 1))
x_test = Numo::NArray[*Datasets::MNIST.new(type: :test).to_a.map(&:pixels)]
y_test_raw = Datasets::MNIST.new(type: :test).to_a.map(&:label)
y_test = Numo::NArray[*y_test_raw].reshape(y_test_raw.size, 1)
puts " Done"

w1, w2 = train(x_train, y_train, x_test, y_test, n_hidden_nodes: 1200, iterations: 1000, lr: 0.3)
