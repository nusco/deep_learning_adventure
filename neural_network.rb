require 'numo/narray'
require 'numo/gsl'

def sigmoid(z)
    1 / (1 + Numo::NMath.exp(-z))
end

def softmax(logits)
    exponentials = Numo::NMath.exp(logits)
    return exponentials / exponentials.sum(axis: 1).reshape(exponentials.size, 1)
end

def sigmoid_gradient(sigmoid)
    sigmoid.dot(1 - sigmoid)
end

def forward(x, w)
    softmax(x)
    sigmoid(x.dot(w))
end

def loss(y, y_hat)
    -(y.dot(Numo::NMath.log(y_hat))).sum() / y.shape[0]
end

def prepend_bias(m)
    m.insert(0, 1, axis: 1)
end

def forward(x, w1, w2)
    h = sigmoid(prepend_bias(x).dot(w1))
    y_hat = softmax(prepend_bias(h).dot(w2))
    [y_hat, h]
end

def back(x, y, y_hat, w2, h)
    w2_gradient = prepend_bias(h).transpose.dot(y_hat - y) / x.shape[0]
    w1_gradient = prepend_bias(x).transpose.dot((y_hat - y).dot(w2[1].transpose)) * sigmoid_gradient(h) / x.shape[0]
    [w1_gradient, w2_gradient]
end

def classify(x, w)
    y_hat, _ = forward(x, w)
    labels = y_hat.max_index(axis: 1)
    labels.reshape(labels.shape[0], 1)
end

def initialize_weights(n_input_variables, n_hidden_nodes, n_classes)
    w1_rows = n_input_variables + 1
    w1 = Numo::DFloat.new(w1_rows, n_hidden_nodes).rand_norm * Numo::NMath.sqrt(1 / w1_rows)

    w2_rows = n_hidden_nodes + 1
    w2 = Numo::DFloat.new(w2_rows, n_classes).rand_norm * Numo::NMath.sqrt(1 / w2_rows)

    [w1, w2]
end

def report(iteration, x_train, y_train, x_test, y_test, w1, w2)
    y_hat, _ = forward(x_train, w1, w2)
    training_loss = loss(y_train, y_hat)
    classifications = classify(x_test, w1, w2)
    accuracy = Numo::GSL::Stats.mean(classifications.eq(y_test)) * 100.0
    puts "Iteration: #{iteration}, Loss: #{loss}, Accuracy: #{accuracy}%"
end

def train(x_train, y_train, x_test, y_test, n_hidden_nodes:, iterations:, lr:)
    n_input_variables = x_train.shape[1]
    n_classes = y_train.shape[1]
    w1, w2 = initialize_weights(n_input_variables, n_hidden_nodes, n_classes)
    iterations.times do |iteration|
        y_hat, h = forward(x_train, w1, w2)
        w1_gradient, w2_gradient = back(x_train, y_train, y_hat, w2, h)
        w1 = w1 - (w1_gradient * lr)
        w2 = w2 - (w2_gradient * lr)
        report(iteration, x_train, y_train, x_test, y_test, w1, w2)
    end
    [w1, w2]
end

def one_hot_encode(labels, number_of_classes: 10)
    result = Numo::Int32.zeros(labels.size, number_of_classes)
    labels.each_with_index {|label, i| result[i, label] = 1 }
    result
end

print "Loading data..."
require 'datasets'
x_train =  Numo::NArray[*Datasets::MNIST.new(type: :train).to_a.map(&:pixels)]
y_train = one_hot_encode(Numo::NArray[*Datasets::MNIST.new(type: :train).to_a.map(&:label)])
x_test = Numo::NArray[*Datasets::MNIST.new(type: :test).to_a.map(&:pixels)]
y_test = Numo::NArray[*Datasets::MNIST.new(type: :test).to_a.map(&:label)]
puts " Done"

w1, w2 = train(x_train, y_train, x_test, y_test, n_hidden_nodes: 1200, iterations: 100, lr: 0.8)
