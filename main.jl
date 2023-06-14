module homo3D
    include("GenerateVoxel.jl")
    include("homo3D.jl")
    include("transform.jl")
    include("visual.jl")
end

using .homo3D

voxel, Density = homo3D.Voxel.GenerateVoxel(40,"topology/grid.txt",0.1)
CH = homo3D.Solver.homo3D(1,1,1,115.4,79.6,voxel);
print(Density, CH)