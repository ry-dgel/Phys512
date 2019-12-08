# Author: Rigel Zifkin
using Plots
using FFTW
using DSP
using Printf
using Statistics: mean

const G = 0.1

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
    t = 0.0
    return Space(dims,dx,bound,lims,rs,t,dt,points)
end

function Space(lims::Array{Float64,1},
               dx::Float64,
               bound::Bool,
               rs::Float64,
               dt::Float64,
               points::Array{Point,1})
    dims = round(Int64,lims ./ dx)
    t = 0.0
    return Space(dims,dx,bound,lims,rs,t,dt,points)
end

# Return an array of each indices position from the center
function rarray(space::Space)
    dims = space.dims
    # Initialize output array
    r = zeros(dims...)
    # Want to loop over cartesian indices of the array to generate the r values
    for i in eachindex(CartesianIndices(r))
        # Calculate the center-zeroed distance of each point in the space
        r[i] = hypot(((Tuple(i) .- dims ./2 .- 0.5).* space.dx)...)
    end
    return r
end

function kernel(space::Space)
    # Generate output array with r values of space
    pot = rarray(space)
    # Soften small values of r
    pot[pot .< space.rs] .= space.rs
    # Calc potential and return
    return 1 ./ pot
end

function density(space::Space)
    # Generate output array
    ρ = zeros(space.dims...)
    # Loop over points, adding their mass to the corresponding grid point.
    for point in space.points
        idx = round.(Int, point.pos ./ space.dx .+ space.dims / 2 .+ 0.5)
        ρ[idx...] += point.m
    end
    return ρ
end

function potential(space::Space)
    return conv(density(space), kernel(space))
end
function potential(space::Space, kernel::Array{Float64,2})
    return conv(density(space), kernel)
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
        output[i][slice(i,1)...] .= (data[slice(i,2)...] .-
                                     data[slice(i,1)...])/dx
        # Last index
        li = s[i]
        output[i][slice(i,li)...] .= (data[slice(i,li)...] .-
                                      data[slice(i,li-1)...])/dx
        for idx in 2:li-1
            output[i][slice(i,idx)...] .= (data[slice(i,idx+1)...] .-
                                           data[slice(i,idx-1)...])/(2*dx)
        end
    end
    return output
end

function updatePoints!(space::Space, force::Array{Array{Float64,2},1})
    for point in space.points
        oldpos = point.pos
        # update point velocity
        point.pos .+= point.v * space.dt
        # Periodic Boundary Condition:
        point.pos .-= round.(point.pos ./ (space.lims)) .* space.lims
        # update point position
        point.v .+= [f[pos2ind(space, oldpos)...] for f in force]/point.m * space.dt
    end
end

function randPoints(xlim::Float64, vlim::Float64, N::Int64, mass::Float64)
    points = [Point(mass, (rand(2) .- 0.5) * xlim, (rand(2) .- 0.5) * vlim)
              for n in 1:N]
    return points
end

function printE(space::Space, pot::Array{Float64,2})
    Ep = potE(space, pot)/length(space.points)
    Ek = mean(kinE.(space.points))
    return Ep, Ek
end

function evolve(space::Space, tmax::Float64)
    kern = kernel(space)
    pot = potential(space, kern)
    iter = 0
    @printf("Iteration %d\n", iter)
    @printf("Energies:\n")
    @printf("\tEk per particle: %f\n", Ek)
    @printf("\tEp per particle: %f\n", Ep)
    @printf("\tTot: %f\n", Ek+Ep)

    while space.t < tmax
        iter += 1
        pot = potential(space, kern)
        force = -gradient(pot, space.dx)
        updatePoints!(space, force)
        space.t += space.dt
        Ep = potE(space, pot)/length(space.points)
        Ek = mean(kinE.(space.points))
        plt = contour(density(space),fill=false)
        @printf("Iteration %d\n", iter)
        @printf("Energies:\n")
        @printf("\tEk: %f\n", Ek)
        @printf("\tEp: %f\n", Ep)
        @printf("\tTot: %f\n", Ek+Ep)
        display(plt)
        sleep(0.1)
    end
end


space = Space([100,100], 0.05, true, 0.5, 0.01, randPoints(5.0,0.0,50,5.0))
evolve(space)
