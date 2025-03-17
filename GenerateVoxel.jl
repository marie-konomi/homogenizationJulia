using Distributed

module Voxel
export GenerateVoxel
using LinearAlgebra
function GenerateVoxel(n, address, radius)
    mass = 1/n
    voxel = zeros(n,n,n)
    voxel_c = zeros(n^3,6)
    p = 0
    for i = 1:n
        for j = 1:n
            for k = 1:n
                p = p + 1
                voxel_c[p,1:3] = [k,j,i]
                voxel_c[p,4:6] = [(k-0.5)*mass,(j-0.5)*mass,(i-0.5)*mass]
            end
        end
    end
    node, strut = ReadStrut(address)

    for i = 1:size(voxel_c, 1)
        for j = 1:length(strut)
            start_n = node[strut[j][1],:][1]
            end_n = node[strut[j][2],:][1]
            center = voxel_c[i,4:6]
            alpha = acosd( round((end_n - start_n)'*(center - start_n) / (norm(center - start_n)*norm(end_n - start_n)), digits=5) )
            beta = acosd( round((start_n - end_n)'*(center - end_n) / (norm(center - end_n)*norm(start_n - end_n)), digits=5) )
            if alpha<90 && beta<90
                distance = norm(cross(end_n - start_n,center - start_n)) / norm(end_n - start_n)
            else
                distance = min(norm(center - start_n),norm(center - end_n))
            end
            if distance<=radius
                
                voxel[Int(voxel_c[i,1]),Int(voxel_c[i,2]),Int(voxel_c[i,3])] = 1
                break
            end
        end
    end
    Density = sum(voxel)/n^3
    return voxel, Density
end

function ReadStrut(address)
    fid = open(address, "r")
    nodelist = []
    strutlist = []
    for line in readlines(fid)
        if line[1] == 'G'
            x = parse(Float64, line[17:24])
            y = parse(Float64, line[25:32])
            z = parse(Float64, line[33:40])
            push!(nodelist, [x, y, z])
        end
        if line[1] == 'S'
            Snode = parse(Int, line[17:24])
            Enode = parse(Int, line[25:32])
            push!(strutlist, [Snode, Enode])
        end
    end
    close(fid)
    return nodelist, strutlist
end
end