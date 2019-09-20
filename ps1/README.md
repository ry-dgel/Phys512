# Assignment 1
## Due 19-08-20 @ 4:00PM

See writeup.pdf for written answers.

derivative.py was used to produce the plot in question 1 of error vs dx size.
it takes no parameters and shows a plot.

interp_lakeshore.py performs volts->temp interpolateion based on data in lakeshore.txt
the script can take any number of input parameters where each value is a voltage.
When given no parameters it produces output for V = 0.69 hardcoded into the file.
I added headers to lakeshore.txt, so please use the copy I provided, cheers.

integrate_vary_step.py will perform integration of 1/(1+x^2) using both the new and old method
the result, error and number of function calls will be printed.

electric_sphere.py will produce (after some time) a plot of the electric field
at a radius r from the center of a spherical shell using both my variable step integrator
and scipy's quad function. It defaults to a radius of 1 and a charge of 10E-6



