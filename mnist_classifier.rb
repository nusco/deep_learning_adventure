require 'numo/narray'
require 'numo/gsl'

def sigmoid(z)
    return 1 / (1 + Numo::NMath.exp(-z))
end

def forward(x, w)
    return sigmoid(x.dot(w))
end

def classify(x, w)
    y_hat = forward(x, w)
    labels = Numo::Int32.zeros(y_hat.shape[0])
    (0...labels.shape[0]).each {|i| labels[i] = y_hat[i, true].max_index }
    labels
end

def loss(x, y, w)
    y_hat = forward(x, w)
    first_term = y * Numo::NMath.log(y_hat)
    second_term = (1 - y) * Numo::NMath.log(1 - y_hat)
    return -(first_term + second_term).sum / x.shape[0]
end

def gradient(x, y, w)
    x.transpose.dot(forward(x, w) - y) / x.shape[0]
end

def report(iteration, x_train, y_train, x_test, y_test, w)
    total_examples = x_test.shape[0]
    matches = (classify(x_test, w).eq y_test).count
    matches_percent = matches * 100 / total_examples
    training_loss = loss(x_train, y_train, w)
    puts "#{iteration} - Loss: #{training_loss}, #{matches_percent}%"
end

def train(x_train, y_train, x_test, y_test, iterations:, lr:)
    w = Numo::Int32.zeros(x_train.shape[1], y_train.shape[1])
    iterations.times do |iteration|
        report(iteration, x_train, y_train, x_test, y_test, w)
        w -= gradient(x_train, y_train, w) * lr
    end
    report(iteration, x_train, y_train, x_test, y_test, w)
    return w
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

w = train(x_train, y_train, x_test, y_test, iterations: 200, lr: 1e-5)
