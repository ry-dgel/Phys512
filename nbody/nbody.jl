# Author: Rigel Zifkin
using Plots
using FFTW
using Printf
using Statistics: mean
using Profile
#using DiffEqOperators

dims = [5.0,5.0]
dx = 5.0/500
npoints = 50000
mass = 5.0
rs = 2*dx
tmax = 5.0
dt = 0.01
const G = 0.001

function ind2pos(space, idx)
    dims = space.dims
    dx = space.dx
    pos = ((idx .- dims ./ 2 .- 0.5) .* dx)
    return pos
end

function pos2ind(space, pos)
    dims = space.dims
    dx = space.dx
    idx = round.(Int, pos ./ dx .+ dims / 2 .+ 0.5)
    return idx
end

struct Point
    m::Float64
    pos::Array{Float64,1}
    v::Array{Float64,1}
end

#=
function potE(point::Point, space, potential)
    return -point.m * potential[pos2ind(space,point.pos)]
end
=#

function kinE(point::Point)
    return 0.5 * point.m * hypot((point.v.^2)...)
end

mutable struct Space
    # The spatial properties of the space
    dims::Array{Int64,1}
    dx::Float64
    bound::Bool
    lims::Array{Float64,1}
    rs::Float64

    # The temporal properties of the space
    t::Float64
    dt::Float64

    # Potential Kernel
    kern::Array{Float64,2}

    # FFT and IFFT operators
    plan
    iplan

    # The points that inhabit the space
    points::Array{Point,1}
end

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
    plan = FFTW.plan_rfft(kern)
    iplan = FFTW.plan_irfft(plan * kern,dims[1])
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

function kernel(dims::Array{Int64,1}, dx::Float64, rs::Float64)
    # Generate output array with r values of space
    pot = rarray(dims, dx)
    # Soften small values of r
    pot[pot .< rs] .= rs
    # Calc potential and return
    return -1 ./ pot
end

function density(space::Space)
    # Generate output array
    ρ = zeros(space.dims...)
    # Loop over points, adding their mass to the corresponding grid point.
    @simd for point in space.points
        @inbounds ρ[pos2ind(space, point.pos)...] += point.m
    end
    return ρ
end

function conv(mat::Array{Float64,2}, kernel::Array{Float64,2}, plan, iplan)
    return iplan*((plan*mat) .* (plan*FFTW.fftshift(kernel)))
end

function potential(space::Space)
    return conv(density(space), space.kern, space.plan, space.iplan)
end

function potE(space::Space, pot::Array{Float64,2})
    return sum([-point.m * pot[pos2ind(space,point.pos)...] for point in space.points])
end
#=
function gradient(data::Array{Float64}, dx::Float64)
    s = size(data)
    d = length(s)
    output = fill(zeros(d),s)
    δs = [zeros(Int64, d) for i in 1:d]
    [δs[i][i] = 1 for i in 1:d]
    @show δs
    for i in eachindex(CartesianIndices(data))
        idx = collect(Tuple(i))
        @show idx
        output[i] = [(data[(idx+δ)...] - data[(idx-δ)...])/(2*dx) for δ in δs]
    end
    for i in d:
        output[i,:]
end
=#

# Useful for taking slices of n-d arrays
function ndcolon(dim,idx,val)
           return vcat(repeat([:], idx-1), [val], repeat([:], dim-idx))
end

function gradient(data::Array{Float64}, dx::Float64)
    s = size(data)
    d = length(s)
    output = Array{Array{Float64, d}, 1}(undef,d)
    for idx in eachindex(output)
        output[idx] = zeros(s...)
    end
    # Take care of edge cases
    slice(idx, val) = ndcolon(d,idx,val)
    for i in 1:d
        # First index
        @views output[i][slice(i,1)...] .= (data[slice(i,2)...] .-
                                            data[slice(i,1)...])/dx
        # Last index
        li = s[i]
        @views output[i][slice(i,li)...] .=  (data[slice(i,li)...] .-
                                              data[slice(i,li-1)...])/dx
        for idx in 2:li-1
            @views output[i][slice(i,idx)...] .= (data[slice(i,idx+1)...] .-
                                                  data[slice(i,idx-1)...])/(2*dx)
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

# Heun's method, much better!#
function updatePoints!(space::Space, force::Array{Array{Float64,2},1})
    @simd for point in space.points
        acc = [f[pos2ind(space, point.pos)...] for f in force]/point.m
        newpos = point.pos + space.dt .* point.v
        newpos .-= round.(newpos ./ (space.lims)) .* space.lims
        acc2 = [f[pos2ind(space, newpos)...] for f in force]/point.m
        # update point position
        point.pos .+= (point.v + space.dt * (acc/3 + acc2/6)) * space.dt
        # Update point velocity
        point.v .+= (acc + acc2) * space.dt / 2
        # Periodic Boundary Condition:
        point.pos .-= round.(point.pos ./ (space.lims)) .* space.lims
    end
end

function randPoints(xlim::Float64, vlim::Float64, N::Int64, mass::Float64)
    points = [Point(mass, (rand(2) .- 0.5) * xlim, (rand(2) .- 0.5) * vlim)
              for n in 1:N]
    return points
end

function calcE(space::Space, pot::Array{Float64,2})
    Ep = potE(space, pot)/length(space.points)
    Ek = mean(kinE.(space.points))
    return Ep, Ek
end

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
            #idxs = collect(1:25:500)
            #xs = repeat(idxs,length(idxs),1)
            #ys = reshape(repeat(idxs,1,length(idxs))',length(idxs)^2,1)
            #u = [force[1][x,y] for x in idxs for y in idxs]
            #v = [force[2][x,y] for x in idxs for y in idxs]
            #plt = quiver(xs,ys,quiver=(u,v))
            #plt = heatmap(force[2])
            plt = heatmap(density(space))
            display(plt)
            #sleep(0.1)
        end
    end
end

#space = Space([50,50], 0.05, true, 0.05, 0.01, randPoints(1.0,0.0,10,5.0))
#evolve(space,0.1, false)
#Profile.clear_malloc_data()
println("Compiling simulation code! First iteration takes a bit to plot due to this.")
#space = Space(dims, dx, true, rs, dt, randPoints(1.0*dims[1],0.0,npoints,mass))
#space = Space([100,100],1/100,true,rs,dt,[Point(1,[0,0],[0,0])])
space = Space([100,100],1/100,true,rs,dt,[Point(1,[0.05,0],[0,0.1]), Point(1,[-0.05,0],[0,-0.1])])
evolve(space, tmax, 10)
