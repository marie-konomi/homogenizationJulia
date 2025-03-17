using Distributed
Distributed.addprocs(6)

@everywhere include("GenerateVoxel.jl")
@everywhere include("homo3D.jl")
@everywhere include("transform.jl")

@everywhere using .Voxel
@everywhere using .Solver

voxel, Density = Voxel.GenerateVoxel(40,"./topology/x_cross_grid.txt",0.1)

E = 0.75e9;
v = 0.3;

lambda = v*E/((1+v) * (1-2*v));
mu = E/2*(1+v);

@time CH = Solver.homo3D(3,3,3,lambda,mu,voxel);
print(Density)
display(CH)