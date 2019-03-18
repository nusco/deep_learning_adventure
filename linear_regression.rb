require 'numo/narray'
require 'numo/gsl'
require 'csv'

def predict(x, w, b)
    x * w + b
end

def loss(x, y, w, b)
    Numo::GSL::Stats.mean((predict(x, w, b) - y) ** 2)
end

def gradient(x, y, w, b)
    w_gradient = Numo::GSL::Stats.mean(2 * X * (predict(x, w, b) - y))
    b_gradient = Numo::GSL::Stats.mean(2 * (predict(x, w, b) - y))
    return w_gradient, b_gradient
end

def train(x, y, iterations:, alpha:)
    w = b = 0
    iterations.times do |iteration|
        puts("#{iteration} => Loss: #{loss(x, y, w, b)}")
        w_gradient, b_gradient = gradient(x, y, w, b)
        w -= w_gradient * alpha
        b -= b_gradient * alpha
    end
    return w, b
end

# Load data
data = CSV.read("pizza.txt", col_sep: "\s", headers: true)
X = Numo::NArray[*data.by_col[0].map(&:to_i)]
Y = Numo::NArray[*data.by_col[1].map(&:to_i)]

# Phase 1: Find the line
w, b = train(X, Y, iterations: 10000, alpha: 0.001)
puts("Parameters: w={w}, b=#{b}")

# Phase 2: Use the line to make a prediction
x = 25
y_hat = predict(x, w, b)
puts("Prediction: reservations=#{x} => pizzas=#{y_hat}")

puts("Enter to continue...")
gets

# Plot chart
require 'matplotlib/pyplot'
plt = Matplotlib::Pyplot
plt.xlabel("Reservations", fontsize: 20)
plt.ylabel("Pizzas", fontsize: 20)
plt.show()
