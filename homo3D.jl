module Solver
export homo3D
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Distributed

function homo3D(lx,ly,lz,lambda,mu,voxel)
    nelx, nely, nelz = size(voxel)
    dx = lx/nelx; dy = ly/nely; dz = lz/nelz
    nel = nelx*nely*nelz
    keLambda, keMu, feLambda, feMu = hexahedron(dx/2,dy/2,dz/2)
    nodenrs = reshape(1:(1+nelx)*(1+nely)*(1+nelz),1+nelx,1+nely,1+nelz)
    edofVec = reshape(1 .+3*nodenrs[1:end-1,1:end-1,1:end-1],nel,1)
    addx = [0 1 2 3*nelx.+[3 4 5 0 1 2] -3 -2 -1]
    addxy = 3*(nely+1)*(nelx+1).+addx
    edof = repeat(edofVec,outer=(1,24)) + repeat([addx addxy],outer=(nel,1))
    nn = (nelx+1)*(nely+1)*(nelz+1)
    nnP = (nelx)*(nely)*(nelz)
    nnPArray = reshape(1:nnP, nelx, nely, nelz)

    nnPArray = cat(nnPArray, reshape(nnPArray[1,:,:],1,size(nnPArray)[2],size(nnPArray)[3]), dims=1)
    nnPArray = cat(nnPArray, reshape(nnPArray[:,1,:],size(nnPArray)[1],1,size(nnPArray)[3]), dims=2)
    nnPArray = cat(nnPArray, reshape(nnPArray[:,:,1],size(nnPArray)[1],size(nnPArray)[2],1), dims=3)
    dofVector = zeros(3*nn, 1)

    dofVector[1:3:end] = 3*nnPArray[:].-2
    dofVector[2:3:end] = 3*nnPArray[:].-1
    dofVector[3:3:end] = 3*nnPArray[:]
    edof = dofVector[edof]
    ndof = 3*nnP
    iK = kron(edof,ones(24,1))'
    jK = kron(edof,ones(1,24))'
    lambda = lambda.*(voxel.==1)
    mu = mu.*(voxel.==1)
    sK = keLambda[:]*lambda[:]' + keMu[:]*mu[:]'
    K = sparse(iK[:], jK[:], sK[:], ndof, ndof)
    K = 1/2*(K+K')
    iF = repeat(edof',outer=(6,1))
    jF = [ones(24,nel); 2*ones(24,nel); 3*ones(24,nel); 4*ones(24,nel); 5*ones(24,nel); 6*ones(24,nel)]
    sF = feLambda[:]*lambda[:]' + feMu[:]*mu[:]'
    F  = sparse(iF[:], jF[:], sF[:], ndof, 6)

    activedofs = edof[findall(voxel[:].== 1), :]
    activedofs = sort(unique(activedofs))
    X = zeros(ndof,6)
    
    #if you want efficient convergence, you needs incomplete Cholesky factorization

    @sync @distributed for i = 1:6
        X[Int.(activedofs)[4:end],i] = cg(K[Int.(activedofs)[4:end],Int.(activedofs)[4:end]], F[Int.(activedofs)[4:end],i], abstol=1e-4, maxiter=300)
    end
    
    X0 = zeros(nel, 24, 6)
    X0_e = zeros(24, 6)
    ke = keMu + keLambda
    fe = feMu + feLambda
    X0_e[4:24,:] = ke[4:24,4:24] \ fe[4:24,:]
    for i = 1:6
        X0[:,:,i] = kron(X0_e[:,i]', ones(nel,1))
    end
    CH = zeros(6,6)
    volume = lx*ly*lz
    for i = 1:6
        @sync @distributed for j = 1:6
            sum_L = ((X0[:,:,i] .- X[Int.(edof .+ (i-1)*ndof)])*keLambda) .* (X0[:,:,j] .- X[Int.(edof .+ (j-1)*ndof)])
            sum_M = ((X0[:,:,i] .- X[Int.(edof .+ (i-1)*ndof)])*keMu) .* (X0[:,:,j] .- X[Int.(edof .+ (j-1)*ndof)])
            sum_L = reshape(sum(sum_L, dims=2), nelx, nely, nelz)
            sum_M = reshape(sum(sum_M, dims=2), nelx, nely, nelz)
            CH[i,j] = round(1/volume*sum(sum(sum(lambda.*sum_L + mu.*sum_M))), digits=4)
        end
    end
    return CH
end

function hexahedron(a, b, c)
    CMu = diagm([2, 2, 2, 1, 1, 1])
    CLambda = zeros(6,6)
    CLambda[1:3,1:3] .= 1
    xx = [-sqrt(3/5), 0, sqrt(3/5)]
    yy = xx
    zz = xx
    ww = [5/9, 8/9, 5/9]
    keLambda = zeros(24,24)
    keMu = zeros(24,24)
    feLambda = zeros(24,6)
    feMu = zeros(24,6)
    for ii = 1:length(xx)
        for jj = 1:length(yy)
            for kk = 1:length(zz)
                x = xx[ii]
                y = yy[jj]
                z = zz[kk]
                qx = [ -((y-1)*(z-1))/8, ((y-1)*(z-1))/8, -((y+1)*(z-1))/8, ((y+1)*(z-1))/8, ((y-1)*(z+1))/8, -((y-1)*(z+1))/8, ((y+1)*(z+1))/8, -((y+1)*(z+1))/8]
                qy = [ -((x-1)*(z-1))/8, ((x+1)*(z-1))/8, -((x+1)*(z-1))/8, ((x-1)*(z-1))/8, ((x-1)*(z+1))/8, -((x+1)*(z+1))/8, ((x+1)*(z+1))/8, -((x-1)*(z+1))/8]
                qz = [ -((x-1)*(y-1))/8, ((x+1)*(y-1))/8, -((x+1)*(y+1))/8, ((x-1)*(y+1))/8, ((x-1)*(y-1))/8, -((x+1)*(y-1))/8, ((x+1)*(y+1))/8, -((x-1)*(y+1))/8]

                J = vcat(qx', qy', qz')*[-a a a -a -a a a -a; -b -b b b -b -b b b; -c -c -c -c c c c c]'
                qxyz = J\vcat(qx', qy', qz')
                B_e = zeros(6,3,8)
                for i_B = 1:8
                    B_e[:,:,i_B] = [qxyz[1,i_B]   0             0; 0             qxyz[2,i_B]   0; 0             0             qxyz[3,i_B]; qxyz[2,i_B]   qxyz[1,i_B]   0; 0             qxyz[3,i_B]   qxyz[2,i_B]; qxyz[3,i_B]   0             qxyz[1,i_B]]
                end
                B = [B_e[:,:,1] B_e[:,:,2] B_e[:,:,3] B_e[:,:,4] B_e[:,:,5] B_e[:,:,6] B_e[:,:,7] B_e[:,:,8]]
                weight = det(J)*ww[ii] * ww[jj] * ww[kk]
                keLambda = keLambda + weight * B' * CLambda * B
                keMu = keMu + weight * B' * CMu * B
                feLambda = feLambda + weight * B' * CLambda
                feMu = feMu + weight * B' * CMu
            end
        end
    end
    return keLambda, keMu, feLambda, feMu
end
end