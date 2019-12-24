Running `julia nbody.jl` will run the code automatically producing gifs of
each step.

Written on julia 1.3 with the FFTW, Plots and Images packaged installed.

Single particle indeed stays motionless.

Two particles kind of orbit, but drift.

k^-3 results are very similar to that of the fully random plots. Both result 
in a large clump with more mass at the center, similar to a small galaxy.

There's some bug that I spent ages trying to track down that results in
everything accelerating down and left. Who knows why...
