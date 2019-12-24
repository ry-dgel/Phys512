# Author: Rigel Zifkin
using Plots
using FFTW
using Printf
using Statistics: mean
using Images

##################
# Sim Parameters #
##################
lims = [200,200]
dx = 5.0/200
npoints = 10000
mass = 5.0
rs = 4*dx
tmax = 20.0
dt = 0.01
const G = 0.0001

# Convert matrix indices to a position in a space
function ind2pos(space, idx)
    dims = space.dims
    dx = space.dx
    pos = ((idx .- dims ./ 2 .- 0.5) .* dx)
    return pos
end

# Convert position in space to matrix indices
function pos2ind(space, pos)
    dims = space.dims
    dx = space.dx
    idx = round.(Int, pos ./ dx .+ dims / 2 .+ 0.5)
    return idx
end

# Structure for a point with mass position and velocity
struct Point
    m::Float64
    pos::Array{Float64,1}
    v::Array{Float64,1}
end

# Calculate the kinetic energy of a point
function kinE(point::Point)
    return 0.5 * point.m * hypot((point.v.^2)...)
end

# Structure for a space in which an nbody simulation take splace
# Also holds any useful objects for simulations in that space
mutable struct Space
    # The spatial properties of the space
    dims::Array{Int64,1} # Raw index i.e position = index * dx
    dx::Float64 # Discretisation of space
    bound::Bool # Wether or not to use periodic boundary conditions
    lims::Array{Float64,1} # Actual dimensional limits of the space i.e dims*dx
    rs::Float64 # Smoothing radius for the potential

    # The temporal properties of the space
    t::Float64  # Total time
    dt::Float64 # Discretisation of time

    # Potential Kernel
    kern::Array{Float64,2}

    # FFT and IFFT operators
    plan
    iplan

    # The points that inhabit the space
    points::Array{Point,1}
end

# Some quick generator functions of spaces
function Space(dims::Array{Int64,1},
               dx::Float64,
               bound::Bool,
               rs::Float64,
               dt::Float64,
               points::Array{Point,1})
    lims = dims .* dx
    return Space(dims,dx,bound,lims,rs,dt,points)
end

function Space(lims::Array{Float64,1},
               dx::Float64,
               bound::Bool,
               rs::Float64,
               dt::Float64,
               points::Array{Point,1})

    dims = round.(Int64,lims ./ dx)
    return Space(dims,dx,bound,lims,rs,dt,points)
end

function Space(dims::Array{Int64,1},
               dx::Float64,
               bound::Bool,
               lims::Array{Float64,1},
               rs::Float64,
               dt::Float64,
               points::Array{Point,1})

    kern = kernel(dims,dx,rs)
    plan = FFTW.plan_fft(kern,[1,2])
    iplan = FFTW.plan_ifft(plan * kern,[1,2])
    t = 0.0
    return Space(dims,dx,bound,lims,rs,t,dt,kern,plan,iplan,points)
end

# Return an array of each indices position from the center
function rarray(dims::Array{Int64,1}, dx::Float64)
    # Initialize output array
    r = zeros(dims...)
    # Want to loop over cartesian indices of the array to generate the r values
    for i in eachindex(CartesianIndices(r))
        # Calculate the center-zeroed distance of each point in the space
        r[i] = hypot(((Tuple(i) .- dims ./2 .- 0.5).* dx)...)
    end
    return r
end

# Calculate an array of k^3 given spatial bounds and Discretisation.
function km3array(dims::Array{Int64,1}, dx::Float64)
    # Calculate fourier limits i.e. Nyquist frequency
    dk = 1/(2*dx)
    ks = zeros(dims...)
    for i in eachindex(CartesianIndices(ks))
        ks[i] = hypot(((Tuple(i) .- dims ./2 .- 0.5).* dk)...)
    end
    # Calculate k^-3 with a small smoothing radius to avoid blowup
    km3 = (ks.+2*dk).^(-3)
    return km3
end

function kernel(dims::Array{Int64,1}, dx::Float64, rs::Float64)
    # Generate output array with r values of space
    pot = rarray(dims, dx)
    # Soften small values of r
    pot[pot .< rs] .= rs
    # Calc potential and return
    return -1 ./ pot
end

# Spreads each point out to 4 nearest grid cells.
# Converts each points position to an index and then adds a quarter of
# the mass to I, I-[1,0], I-[0,1], I-[1,1]
# TODO: Non-Periodic Boundary
function density(space::Space, bound::Bool=false)
    d = space.dims
    # Generate output array
    ρ = zeros(d...)
    # Loop over points, adding their mass to the corresponding grid point.
    @simd for point in space.points
        ind = pos2ind(space, point.pos)
        if ind == [1,1]
            inds = [[1,1],[d[1],1],[1,d[2]],[d[1],d[2]]]
        elseif ind[1] == 1
            inds = [ind,ind.-[0,1],[d[1],ind[2]],[d[1],ind[2]-1]]
        elseif ind[2] == 1
            inds = [ind,ind.-[1,0],[ind[1],d[2]],[ind[1]-1,d[2]]]
        elseif ind == [1,1]
            inds = [[1,1],[d[1],1],[1,d[2]],[d[1],d[2]]]
        else
            inds = [ind,ind.-[1,0],ind.-[0,1],ind.-[1,1]]
        end
        for i in inds
            @inbounds ρ[i...] += point.m/4.0
        end
    end
    return ρ
end

# Performs a convolution between a matrix and a kernel given an 
# planned fft and ifft operator.
# TODO: Non-Periodic Boundary
function conv(mat::Array{Float64,2}, kernel::Array{Float64,2}, plan, iplan)
    return real.(iplan*((plan*mat) .* (plan*FFTW.fftshift(kernel))))
end

# Computes the gravitational potential of a space via convolution
function potential(space::Space)
    return conv(density(space), space.kern, space.plan, space.iplan)
end

# Calculate the total potential energy in a space.
function potE(space::Space, pot::Array{Float64,2})
    return sum([-point.m * pot[pos2ind(space,point.pos)...] for point in space.points])
end

# Useful for taking slices of n-d arrays
function ndcolon(dim,idx,val)
           return vcat(repeat([:], idx-1), [val], repeat([:], dim-idx))
end

# Arbitrary dimension gradient function.
# Returns a list of n-d arrays that correspond to the derivative along the ith
# axis. 
# Uses padded arrays to allow for different boundary conditions
function gradient(data::Array{Float64}, dx::Float64, bound::Bool=false)
    if bound
        padding = Fill(0,(1,1),(1,1))
    else
        padding = Pad(:circular,1,1)
    end
    padded = padarray(data, padding)
    s = size(data)
    d = length(s)
    output = Array{Array{Float64, d}, 1}(undef,d)
    for idx in eachindex(output)
        output[idx] = zeros(s...)
    end
    slice(idx, val) = ndcolon(d,idx,val)
    for i in 1:d
        li = s[i]
        for idx in 1:li
            @views output[i][slice(i,idx)...] .= ((padded[slice(i,idx+1)...] .-
                                                   padded[slice(i,idx-1)...])/(2*dx))[1:li]
        end
    end
    return output
end

#= 'Euler' method of update, not great
function updatePoints!(space::Space, force::Array{Array{Float64,2},1})
    @simd for point in space.points
        oldpos = point.pos
        # update point velocity
        point.pos .+= point.v * space.dt
        # Periodic Boundary Condition:
        point.pos .-= round.(point.pos ./ (space.lims)) .* space.lims
        # update point position
        point.v .+= [f[pos2ind(space, oldpos)...] for f in force]/point.m * space.dt
    end
end
=#

# Heun's method of position and velocity update
# Still shows bad signs of energy non-conservation, but neither decays
# nor explodes...
# TODO: Non-Periodic Boundary
function updatePoints!(space::Space, force::Array{Array{Float64,2},1})
    @simd for point in space.points
        # Compute first guess at acceleration
        acc = [f[pos2ind(space, point.pos)...] for f in force]/point.m
        # Calculate new position with current velocity
        newpos = point.pos + space.dt .* point.v
        # Wrap new position for periodic boundary conditions
        newpos .= newpos .- round.(newpos ./ (space.lims)) .* space.lims
        # Calculate acceleration at new position
        acc2 = [f[pos2ind(space, newpos)...] for f in force]/point.m
        # Update point position using average of accelerations
        point.pos .= point.pos .+ (point.v + space.dt * (acc/3 + acc2/6)) * space.dt
        # Update point velocity using average of accelerations
        point.v .= point.v .+ (acc + acc2) * space.dt / 2
        # Periodic Boundary Condition:
        point.pos .= point.pos -  round.(point.pos ./ (space.lims)) .* space.lims
    end
end

# Generate a set of N random points with position within [xlim,xlim]
# and velocity within [vlim,vlim], All points have same mass
function randPoints(xlim::Float64, vlim::Float64, N::Int64, mass::Float64)
    points = [Point(mass, (rand(2) .- 0.5) * xlim, (rand(2) .- 0.5) * vlim)
              for n in 1:N]
    return points
end

# Generate a set of N random points with position within [xlim, xlim]
# and velocity within [vlim, vlim], Mass is distributed with spatial 
# fluctuations generated from a k^-3 power law relation.
function randkm3Points(xlim::Float64, 
                       vlim::Float64, 
                       N::Int64, 
                       mass_scale::Float64,
                       space::Space)
    # Generate output list
    points = Array{Point,1}()
    # Make mass fluctuations based on k^-3 power spectrum
    km3 = km3array(space.dims, space.dx)
    mass_flucts = abs.(ifftshift(space.iplan * km3)) # ifft to get proper distribution
    
    # Generate the points and push to array
    for n in 1:N
        pos = (rand(2) .- 0.5) * xlim
        vel = (rand(2) .- 0.5) * vlim
        mass = mass_scale * (1+mass_flucts[pos2ind(space, pos)...])
        push!(points, Point(mass, pos, vel))
    end
    return points
end

# Calculate the total energy in a space with a given potential.
function calcE(space::Space, pot::Array{Float64,2})
    Ep = potE(space, pot)/length(space.points)
    Ek = mean(kinE.(space.points))
    return Ep, Ek
end

# Run a simulation up until tmax, spits out a plot after
# genPlots number of frames if genPlots is greater than 0.
function evolve(space::Space, tmax::Float64, genPlots::Int)
    pot = potential(space)
    iter = 0
    while space.t < tmax
        Ep, Ek = calcE(space, pot)
        @printf("Iteration %d\n", iter)
        @printf("Energies:\n")
        @printf("\tEk: %f\n", Ek)
        @printf("\tEp: %f\n", Ep)
        @printf("\tTot: %f\n", Ek+Ep)

        iter += 1
        pot = potential(space)
        force = -G * gradient(pot, space.dx)
        updatePoints!(space, force)
        space.t += space.dt
        if genPlots > 0 && (iter % genPlots == 0)
            plt = heatmap(density(space))
            display(plt)
        end
    end
end

# Same as above, but uses the plots to generate a gif that is saved
# to "name.gif" at the end.
function plot_evolve(space::Space, tmax::Float64, frame_every::Int, name::String)
    pot = potential(space)
    plt = heatmap(density(space))
    max_frame = tmax / space.dt
    iter = 1
    anim = @animate for iter=1:max_frame
        Ep, Ek = calcE(space, pot)
        @printf("Iteration %d\n", iter)
        @printf("Energies:\n")
        @printf("\tEk: %f\n", Ek)
        @printf("\tEp: %f\n", Ep)
        @printf("\tTot: %f\n", Ek+Ep)

        pot = potential(space)
        force = -G * gradient(pot, space.dx)
        updatePoints!(space, force)
        space.t += space.dt
        
        # Doesn't overlay but just appends data? Plotting
        # seems to get really slow at high iteration numbers.
        if (iter % frame_every == 0)
            heatmap!(plt[1],density(space))
        end
    end every frame_every
 
    gif(anim,"$name.gif", fps=24)
    return anim
end

println("Compiling simulation code! First iteration takes a bit to plot due to this.")
space = Space([51,51],dx,true,rs,dt,[Point(1,[0,0],[0,0])])
println("Running Single point simulation.")
#plot_evolve(space, tmax/3, 10, "Single")

println("Running two point orbit simulation.")
# Needs more tweaking to get particles to actually orbit, why do they drift??
space = Space([100,100],dx,true,rs,dt,[Point(100,[0.01,0],[0,0.01]), Point(100,[-0.01,0],[0,-0.01])])
plot_evolve(space, tmax, 10, "Orbit")

println("Running $npoints point dense simulation.")
space = Space(lims, dx, true, rs, dt, randPoints(1.0*lims[1]*dx,0.0,npoints,mass))
plot_evolve(space, tmax, 10, "Dense")

println("Running $npoints point k^-3 simulation.")
space = Space(lims, dx, true, rs, dt, randkm3Points(1.0*lims[1]*dx, 0.0, npoints, 5.0, space))
#plot_evolve(space, tmax, 10,"km3")
