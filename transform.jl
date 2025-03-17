using Distributed
using LinearAlgebra

function transform(itr, tmx)
    ne = length(itr)
    nd = ndims(itr)
    if ne == 3
        nd = 1
    end
    otr = copy(itr)
    otr .= 0
    iie = zeros(Int, nd)
    ioe = zeros(Int, nd)
    cne = cumprod(3*ones(nd))/3
    for oe = 1:ne
        ioe = mod.(div.(oe .- 1, cne), 3) .+ 1
        for ie = 1:ne
            pmx = 1
            iie = mod.(div.(ie .- 1, cne), 3) .+ 1
            for id = 1:nd
                pmx *= tmx[ioe[id], iie[id]]
            end
            otr[oe] += pmx * itr[ie]
        end
    end
    return otr
end
