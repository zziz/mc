import tensorflow as tf

# Basic idea 
# Sample a bunch of points from uniform distribution in R2 into a square
# Draw a circle inside this square and count points falling into this square
# Area of circle is S = pi * r**2 => pi = S / r**2, since r = 1.0 => pi = S (area of circle)
# Area of circle is ratio of points falling inside circle multiplied to area of square (which is 4)
# For more detail, check https://en.wikipedia.org/wiki/Monte_Carlo_integration 

pi_true =  3.1415926535897932384626433   # true value of pi

points_n = 1000000 
radius = 1.0
area = (2 * radius) ** 2

points = tf.random_uniform([points_n, 2], minval = -1, maxval = 1)  # random point coordinates sampled from uniform distribution
distance = tf.norm(points, axis = 1)                                # distance to each point
in_circle = tf.less(distance, 1)                                    # points that lie inside circle 
points_in_circle = tf.count_nonzero(in_circle)                      # number of points that lie inside circle 
pi = 4 * (points_in_circle / points_n)                              # value of pi equals to four points that lie inside circle to all samples
sess = tf.Session()  
pi_estimate = sess.run(pi)

print('Pi estimate: {}'.format(pi_estimate))
print('Accuracy ({} samples): {} %'.format(points_n, 100 * abs(pi_true - pi_estimate) / pi_true))
