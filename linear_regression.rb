require 'numo/narray'
require 'numo/gsl'
require 'csv'

def predict(x, a, b)
    a * x + b
end

def loss(x, y, a, b)
    squared_errors = (predict(x, a, b) - y) ** 2
    Numo::GSL::Stats.mean(squared_errors)
end

def gradient(x, y, a, b)
    w_gradient = Numo::GSL::Stats.mean(2 * X * (predict(x, a, b) - y))
    b_gradient = Numo::GSL::Stats.mean(2 * (predict(x, a, b) - y))
    return w_gradient, b_gradient
end

def train(x, y, iterations:, lr:)
    a = b = 0
    iterations.times do |iteration|
        puts "#{iteration} => Loss: #{loss(x, y, a, b)}"
        a_gradient, b_gradient = gradient(x, y, a, b)
        a -= a_gradient * lr
        b -= b_gradient * lr
    end
    return a, b
end

# Load data
data = CSV.read("pizza.txt", col_sep: "\s", headers: true)
X = Numo::NArray[*data['Reservations'].map(&:to_f)]
Y = Numo::NArray[*data['Pizzas'].map(&:to_i)]

# Phase 1: Find the line
a, b = train(X, Y, iterations: 10000, lr: 0.001)
puts "Parameters: a=#{a}, b=#{b}"

# Phase 2: Use the line to make a prediction
reservations = 25
pizzas = predict(reservations, a, b)
puts "Prediction: reservations=#{reservations} => pizzas=#{pizzas}"

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
plt.plot([0, x_edge], [b, predict(x_edge, a, b)], linewidth=2.0, color="b")
plt.show()
