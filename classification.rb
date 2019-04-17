require 'numo/narray'
require 'numo/gsl'
require "numo/linalg"
require 'csv'

def sigmoid(z)
    1 / (1 + Numo::DFloat::Math.exp(-z))
end

def predict(x, w)
    sigmoid(Numo::Linalg.matmul(x, w))
end

def classify(x, w)
    predict(x, w).round()
end

def loss(x, y, w)
    y_hat = predict(X, w)
    first_term = y * Numo::DFloat::Math.log(y_hat)
    second_term = (1 - Y) * Numo::DFloat::Math.log(1 - y_hat)
    -Numo::GSL::Stats.mean(first_term + second_term)
end

def gradient(x, y, w)
    Numo::Linalg.matmul(x.transpose, (predict(x, w) - y)) / x.shape[0]
end

def train(x, y, iterations:, lr:)
    w = Numo::DFloat.zeros(X.shape[1], 1)
    iterations.times do |iteration|
        puts "#{iteration} => Loss: #{loss(x, y, w)}"
        w -= gradient(x, y, w) * lr
    end
    w
end

def test(x, y, w)
    total_examples = x.shape[0]
    matches = (classify(x, w).eq y).count
    matches_percent = matches * 100.0 / total_examples
    puts "Success: #{matches}/#{total_examples}"
end

# Load data
data = CSV.read("break_even.txt", col_sep: "\s", headers: true)
X = Numo::NArray.column_stack([
      Numo::DFloat.ones(data.size),
      data['Reservations'].map(&:to_f),
      data['Temperature'].map(&:to_f),
      data['Tourists'].map(&:to_f)
    ])
Y = Numo::NArray[*data['Break-even'].map(&:to_i)].reshape(data.size, 1)

w = train(X, Y, iterations: 10000, lr: 0.001)
test(X, Y, w)
