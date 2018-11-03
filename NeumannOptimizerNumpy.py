import numpy as np
import time
from math import exp
import matplotlib.pyplot as plt

def gradient_descent( func, initial_x, eps=1e-5, maximum_iterations=65536, learning_rate=1e-2 ):
    """
    Gradient Descent
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    initial_x:          the starting point, should be a float
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """

    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix(initial_x)

    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0

    # gradient updates
    while True:

        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.matrix( gradient )

        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        direction = -gradient

        if np.linalg.norm(direction)<eps:
            break

        t = learning_rate

        x = x + t * direction

        iterations += 1
        if iterations >= maximum_iterations:
            break
    return (x, values, runtimes, xs)

def linear_regression(x, y, w, b, order=0):
    output = w*x.T + b
    error = np.mean((y-output)**2)
    if order == 1:
        grad_w = -2*x.T*(y-(w*x.T + b))
        grad_b = -2*(y-(w*x.T + b))
        grad_w = np.mean(grad_w)
        grad_b = np.mean(grad_b)
        return output, grad_w, grad_b
    return output

def boyd_example_func(x, order=0):
  a=np.matrix('1  3')
  b=np.matrix('1  -3')
  c=np.matrix('-1  0')
  x=np.asmatrix(x)

  value = exp(a*x-0.1)+exp(b*x-0.1)+exp(c*x-0.1)
  if order==0:
      return value
  elif order==1:
      gradient = a.T*exp(a*x-0.1)+b.T*exp(b*x-0.1)+c.T*exp(c*x-0.1)
      return (value, gradient)
  elif order==2:
      gradient = a.T*exp(a*x-0.1)+b.T*exp(b*x-0.1)+c.T*exp(c*x-0.1)
      hessian = a.T*a*exp(a*x-0.1)+b.T*b*exp(b*x-0.1)+c.T*c*exp(c*x-0.1)
      return (value, gradient, hessian)
  else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")

def neumann( func, initial_x, learning_rate=1e-2, eps=1e-5, maximum_iterations=65536):
    x = np.matrix(initial_x)
    # moving_average = x
    neumann_iterate = 0
    iterate = 0
    k_value = 10
    values = []
    runtimes = []
    xs = []
    grad_norm = []
    start_time = time.time()
    while True:
        print(x)
        if iterate < 5:
            value, grad = func(x, 1)
            x = x - learning_rate*grad
            iterate += 1
            continue

        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        eta = 0.5/iterate
        mu = iterate/(iterate + 1)
        mu = min(max(mu, 0.5),0.9)

        value, grad = func(x, 1)

        grad_norm.append(np.linalg.norm(grad)**2)

        if np.linalg.norm(grad)**2 < eps:
            break

        if iterate % k_value == 0:
            neumann_iterate = -eta*grad
            k_value *= 2

        #Removing crazy function as we're only trying on convex function

        neumann_iterate = mu*neumann_iterate - eta*grad

        x = x + mu*neumann_iterate - eta*grad
        # moving_average =
        iterate += 1
        if iterate >= maximum_iterations:
            break
    return x,values,runtimes,xs,grad_norm


def draw_contour( func, neumann_xs, fig, levels=np.arange(5, 1000, 10), x=np.arange(-5, 5.1, 0.05), y=np.arange(-5, 5.1, 0.05)):
    """
    Draws a contour plot of given iterations for a function
    func:       the contour levels will be drawn based on the values of func
    gd_xs:      gradient descent iterates
    newton_xs:  Newton iterates
    fig:        figure index
    levels:     levels of the contour plot
    x:          x coordinates to evaluate func and draw the plot
    y:          y coordinates to evaluate func and draw the plot
    """
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = func( np.matrix([x[i],y[j]]).T , 0 )

    plt.figure(fig)
    plt.contour( x, y, Z.T, levels, colors='0.75')
    plt.ion()
    plt.show()

    # line_gd, = plt.plot( gd_xs[0][0,0], gd_xs[0][1,0], linewidth=2, color='r', marker='o', label='GD' )
    line_newton, = plt.plot( neumann_xs[0][0,0], neumann_xs[0][1,0], linewidth=2, color='m', marker='o',label='Neumann' )

    L = plt.legend(handles=[line_newton])
    plt.draw()
    time.sleep(1)

    for i in range( 1, len(neumann_xs)):

        # line_gd.set_xdata( np.append( line_gd.get_xdata(), gd_xs[ min(i,len(gd_xs)-1) ][0,0] ) )
        # line_gd.set_ydata( np.append( line_gd.get_ydata(), gd_xs[ min(i,len(gd_xs)-1) ][1,0] ) )

        line_newton.set_xdata( np.append( line_newton.get_xdata(), neumann_xs[ min(i,len(neumann_xs)-1) ][0,0] ) )
        line_newton.set_ydata( np.append( line_newton.get_ydata(), neumann_xs[ min(i,len(neumann_xs)-1) ][1,0] ) )


        # L.get_texts()[0].set_text( " GD, %d iterations" % min(i,len(gd_xs)-1) )
        L.get_texts()[0].set_text( " Neumann, %d iterations" % min(i,len(neumann_xs)-1) )

        plt.draw()
        input("Press Enter to continue...")


initial_x = np.matrix('-1.0; -1.0')

x, values, runtimes, neumann_xs, grad_norm = neumann(boyd_example_func, initial_x)
x_gd, gd_values, runtimes_gd, gradient_xs = gradient_descent(boyd_example_func, initial_x)
plt.figure(1)
line_gd, = plt.semilogy([x for x in values], linewidth=2, color='r', marker='o', label='Neumann')
line_neumann, = plt.semilogy([x for x in gd_values], linewidth=2, color='b', marker='o', label='Neumann')
plt.figure(2)
plt.semilogy([x for x in grad_norm], linewidth=2, color='b', marker='o', label='Neumann')
draw_contour( boyd_example_func, neumann_xs, 3, levels=np.arange(0, 15, 1), x=np.arange(-2, 2, 0.1), y=np.arange(-2, 2, 0.1))
