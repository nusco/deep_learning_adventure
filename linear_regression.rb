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

def train(x, y, iterations:, lr:)
    w = b = 0
    iterations.times do |iteration|
        puts "#{iteration} => Loss: #{loss(x, y, w, b)}"
        w_gradient, b_gradient = gradient(x, y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    end
    return w, b
end

# Load data
data = CSV.read("pizza.txt", col_sep: "\s", headers: true)
X = Numo::NArray[*data['Reservations'].map(&:to_f)]
Y = Numo::NArray[*data['Pizzas'].map(&:to_i)]

# Phase 1: Find the line
w, b = train(X, Y, iterations: 10000, lr: 0.001)
puts "Parameters: w={w}, b=#{b}"

# Phase 2: Use the line to make a prediction
x = 25
y_hat = predict(x, w, b)
puts "Prediction: reservations=#{x} => pizzas=#{y_hat}"

puts "Enter to continue..."
gets

# Plot chart
require 'matplotlib/pyplot'
plt = Matplotlib::Pyplot
plt.xlabel("Reservations", fontsize: 20)
plt.ylabel("Pizzas", fontsize: 20)
plt.plot(X.to_a, Y.to_a, "go")
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=2.0, color="b")
plt.show()
