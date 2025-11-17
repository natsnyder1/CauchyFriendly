#pragma once

#include <vector>
#include <cstddef>

using Scalar = double; // TODO: replace by high-precision from boost/GMP

struct Arrangement {
    // d x n matrix: rows = dimension, cols = hyperplanes
    std::size_t dim;               // d
    std::size_t n;                 // number of hyperplanes
    std::vector<Scalar> data;      // row-major: data[row * n + col]

    Arrangement() : dim(0), n(0) {}
    Arrangement(std::size_t d, std::size_t num)
        : dim(d), n(num), data(d * num, Scalar(0)) {}

    Scalar& operator()(std::size_t row, std::size_t col) {
        return data[row * n + col];
    }
    const Scalar& operator()(std::size_t row, std::size_t col) const {
        return data[row * n + col];
    }
};

// Top-level API
int count_chambers(const Arrangement& A);



// module CountingChambers
// using GAP
// using Nemo

// include("examples.jl")

// function __init__()
// GAP.Globals.LoadPackage(GapObj("images"))
// end

// import Base.copy

// export characteristic_polynomial, number_of_chambers, whitney_numbers, betti_numbers, pseudo_minimal_image, canonical_image, minimal_image, trivial_minimal_image

// export threshold_hyperplanes, resonance_hyperplanes, symmetry_threshold, symmetry_resonance
// export dodecahedron_hyperplanes, symmetry_dodecahedron, icosahedron_hyperplanes, symmetry_icosahedron, regular_24_cell_hyperplanes, symmetry_24_cell
// export regular_600_cell_hyperplanes, symmetry_600_cell, regular_120_cell_hyperplanes, symmetry_120_cell
// export crosspolytope_hyperplanes, symmetry_crosspolytope, demicube_hyperplanes, symmetry_demicube, permutohedron_hyperplanes, symmetry_permutohedron
// export discriminantal_hyperplanes, symmetry_discriminantal, soft_discriminantal_hyperplanes, symmetry_soft_discriminantal


// #####################################################################

// ###########################
// #### Structures
// ###########################

// mutable struct Node{HT<:Integer}
//     hashed_rest::HT
//     multiplicity::Int64
// end

// mutable struct Hyperplane_Tree{T <: Union{Nemo.RingElem,Integer}, HT<:Integer}
// 	Hyperplanes::Union{Vector{MatElem},Vector{Matrix{T}}}#one entry for every thread
// 	hash_type::Union{Type{Int64},Type{Int128},Type{BigInt}}
// 	hash_base::Int64
// 	TmpHyperplanes::Union{Vector{MatElem},Vector{Matrix{T}}}#one entry for every thread
// 	TmpRestArray::Array{Array{Int64,1},1}#one array for every thread
// 	L::Array{Dict{HT,Node{HT}}}#The leaves dictionary, keys are (perfectly) hashed arrays
// 	Betti::Array{Array{Int64,1},1}#one entry for every thread
//     GensDict::Array{Dict{Int64,Array{Array{Int64,1},1}},1}#one entry for every thread
//     StabilizerDict::Dict{Int64,GAP.GapObj}
//     I_array_cache::Dict{Int64,Dict{Int64, Array{Int64}}}
//     J_array_cache::Dict{Int64,Dict{Int64, Array{Int64}}}
//     nemo_tmp1::Dict{Int64,T}
//     nemo_tmp2::Dict{Int64,T}
//     DictLockArray::Array{ReentrantLock,1}
// end

// ###########################
// #### Constructors
// ###########################

// function Hyperplane_Tree(Hyperplanes; verbose=false)
// 	m,n = size(Hyperplanes)
// 	T = typeof(Hyperplanes[1,1])
// 	hash_base = Int(ceil(log(2,n)))
// 	max_bits = hash_base*(m - 2)
// 	if max_bits < 64
// 		hash_type = Int64
// 	elseif max_bits < 128
// 		hash_type = Int128
// 	else
// 		hash_type = BigInt
// 	end
// 	verbose && println("Hash type: ", hash_type)
// 	verbose && println("Hash base: ", hash_base)

// 	HyperplanesArray = Array{Array{T,2},1}()
// 	TmpHyperplanesArray = Array{Array{T,2},1}()
// 	if Hyperplanes isa MatElem
// 		HyperplanesArray = Array{MatElem,1}()
// 		TmpHyperplanesArray = Array{MatElem,1}()
// 	end
// 	TmpRestArray = Array{Array{Int64,1},1}()

// 	for id in 1:Threads.nthreads()
// 		push!(HyperplanesArray, deepcopy(Hyperplanes))
// 		push!(TmpHyperplanesArray, deepcopy(Hyperplanes))
// 		push!(TmpRestArray,zeros(Int64,m))
// 	end

//     RootNode = Node{hash_type}(0,1)
//     AllLeaves = Array{Dict{hash_type,Node{hash_type}},1}()
//     for i in 1:n+1
//     	AllLeaves = push!(AllLeaves, Dict{hash_type,Node{hash_type}}())
// 	end
// 	AllLeaves[1][0] = RootNode
// 	Betti = Array{Array{Int64,1},1}()
// 	for id in 1:Threads.nthreads()
// 		push!(Betti, zeros(Int64, m+1))
// 	end

//     GensDict = Array{Dict{Int64,Array{Array{Int64,1},1}},1}()
//     StabilizerDict = Dict{Int64,GAP.GapObj}()

//     I_array_cache = Dict{Int64,Dict{Int, Array{Int64}}}()
//     J_array_cache = Dict{Int64,Dict{Int, Array{Int64}}}()

//     type = typeof(Hyperplanes[1,1])
//     ring = parent(Hyperplanes[1,1])

//     nemo_tmp1 = Dict{Int64,type}()
//     nemo_tmp2 = Dict{Int64,type}()
//     for id in 1:Threads.nthreads()
//         I_array_cache[id] = Dict{Int64,Dict{Int, Array{Int64}}}()
//         J_array_cache[id] = Dict{Int64,Dict{Int, Array{Int64}}}()
//         for i in 0:m
//             I_array_cache[id][i] = zeros(Int64, i)
//             J_array_cache[id][i] = zeros(Int64, i)
//         end
//         nemo_tmp1[id] = ring()
//         nemo_tmp2[id] = ring()
//     end

//     #Create an array of locks, one for each depth so that not the entire dict is locked all the time.
// 	DictLockArray = Array{ReentrantLock,1}()
// 	for i in 1:n
// 		push!(DictLockArray, ReentrantLock())
// 	end

// 	return Hyperplane_Tree(HyperplanesArray, hash_type, hash_base, TmpHyperplanesArray, TmpRestArray, AllLeaves, Betti,
// 		GensDict, StabilizerDict, I_array_cache, J_array_cache, nemo_tmp1, nemo_tmp2, DictLockArray)
// end

// function compute_group_data!(T::Hyperplane_Tree, SymmetryGroup::Union{GAP.GapObj,Array{Array{Int64,1},1},Nothing}, proportion, max_size, min_size, verbose)
//     m,n = size(T.Hyperplanes[1])
//     #Produce necessary group theory data
//     GensDict = Dict{Int64,Array{Array{Int64,1},1}}()

//     G = SymmetryGroup
// 	if typeof(SymmetryGroup) == Array{Array{Int64,1},1}
// 		perms = []
// 		for I in SymmetryGroup
//     		push!(perms, GAP.Globals.PermList(GapObj(I)))
//        	end
// 		G = GAP.Globals.Group(GapObj(perms))
// 	end

//     for depth in 2:n-1
//         H = get_stabilizer(depth, n, G)
//         T.StabilizerDict[depth] = H

//         Gens = get_stoch_gens(H, n, proportion, max_size, min_size)
//         GensDict[depth] = Gens
//         verbose && println("Depth: ", depth, " Stab Size: ", GAP.Globals.Size(H), " Number of gens: ", length(Gens))
//     end
//     for id in 1:Threads.nthreads()
//     	push!(T.GensDict,GensDict)
//     	GensDict = deepcopy(GensDict)
//     end
// end

// ###########################
// #### Data analysis
// ###########################

// function data_average(D)
// 	n=length(D)
// 	Aleaves_at_depth=sum([d.leaves_at_depth for d in D])/n
// 	Atotal_leaves_at_depth=sum([d.total_leaves_at_depth for d in D])/n
// 	Abetti_at_depth=sum([d.betti_at_depth for d in D])/n
// 	Atime_per_depth=sum([d.time_per_depth for d in D])/n
// 	Amem_per_depth=sum([d.mem_per_depth for d in D])/n
// 	Agctime_per_depth=sum([d.gctime_per_depth for d in D])/n
// 	Abytes_per_depth=sum([d.bytes_per_depth for d in D])/n
// 	A=datastruct(Aleaves_at_depth,Atotal_leaves_at_depth,Abetti_at_depth,Atime_per_depth,Amem_per_depth,Agctime_per_depth,Abytes_per_depth)
// 	return A
// end

// ###########################
// #### Main Functions
// ###########################

// function characteristic_polynomial(H::Union{Array{T,2},MatElem}; ConstantTerms::Union{Vector{T},Nothing}=nothing, SymmetryGroup::Union{GAP.GapObj,Array{Array{Int64,1},1},Nothing}=nothing,
// 	OrbitRepresentation=pseudo_minimal_image, proportion=0.01, max_size=nothing, min_size=nothing, multi_threaded=false, verbose=false) where {T <: Union{Nemo.RingElem,Integer}}

// 	BettiNumbers = betti_numbers(H, ConstantTerms=ConstantTerms, SymmetryGroup=SymmetryGroup, OrbitRepresentation=OrbitRepresentation,
//     	proportion=proportion, max_size=max_size, min_size=min_size, multi_threaded=multi_threaded, verbose=verbose)

// 	R, t = polynomial_ring(ZZ, "t")
// 	chi = R(0)
// 	n = length(BettiNumbers)
// 	for i in 1:n
// 		chi += (-1)^(i+1)*BettiNumbers[i]*t^(n-i)
// 	end
// 	return chi
// end

// function number_of_chambers(H::Union{Array{T,2},MatElem}; ConstantTerms::Union{Vector{T},Nothing}=nothing, SymmetryGroup::Union{GAP.GapObj,Array{Array{Int64,1},1},Nothing}=nothing,
// 	OrbitRepresentation=pseudo_minimal_image, proportion=0.01, max_size=nothing, min_size=nothing, multi_threaded=false, verbose=false) where {T <: Union{Nemo.RingElem,Integer}}

//     return sum(betti_numbers(H, ConstantTerms=ConstantTerms, SymmetryGroup=SymmetryGroup, OrbitRepresentation=OrbitRepresentation,
//     	proportion=proportion, max_size=max_size, min_size=min_size, multi_threaded=multi_threaded, verbose=verbose))
// end

// function betti_numbers(H::Matrix{T}, characteristic::Int) where T <: Integer
// 	@assert(isprime(characteristic), "characteristic must be a prime number")
// 	M = matrix(GF(characteristic), H)
// 	return betti_numbers(M)
// end

// function betti_numbers(H::Union{MatElem,Matrix{Int},Matrix{SafeInt64},Matrix{RingElem}}; ConstantTerms::Union{Vector{T},Nothing}=nothing, SymmetryGroup::Union{GAP.GapObj,Array{Array{Int64,1},1},Nothing}=nothing,
//     OrbitRepresentation=pseudo_minimal_image, proportion=0.01, max_size=nothing, min_size=nothing, multi_threaded=false, verbose=false) where T <: Union{Integer,RingElem}

// 	# Treat the trivial case of one-dimensional arrangements separatly.
//     if size(H,1) == 1 && ConstantTerms == nothing
//         return [1,1]
//     end

// 	Arrangement = H
// 	if ConstantTerms != nothing
// 		#Compute chambers of the coned arrangement and half the number of chambers
//     	Arrangement = coned_arrangement(H, ConstantTerms)
//     end


// 	HT=Hyperplane_Tree(Arrangement; verbose=verbose)
// 	n = size(H,2)

// 	if SymmetryGroup != nothing
// 		actual_max_size = 2*n
// 		if max_size != nothing
// 			actual_max_size = max_size
// 		end
// 		actual_min_size = n
// 		if min_size != nothing
// 			actual_min_size = min_size
// 		end
//     	compute_group_data!(HT, SymmetryGroup, proportion, actual_max_size, actual_min_size, verbose)
// 	else
//         OrbitRepresentation = trivial_minimal_image
//     end

// 	delete_restrict_all!(HT, OrbitRepresentation, multi_threaded, verbose)
// 	BettiNumbers = sum(HT.Betti)

// 	if ConstantTerms != nothing
// 		DeConedBettiNumbers = Int64[]
//     	push!(DeConedBettiNumbers, 1)
//     	for i in 1:(length(BettiNumbers)-2)
//         	push!(DeConedBettiNumbers, BettiNumbers[i+1] - DeConedBettiNumbers[i])
//     	end
//     	return DeConedBettiNumbers
//     else
// 		return BettiNumbers
// 	end
// end

// const whitney_numbers = betti_numbers

// function delete_restrict_all!(T::Hyperplane_Tree, OrbitRepresentation, multi_threaded, verbose)

//     for i in find_depth(T):size(T.Hyperplanes[1],2)-1
//         verbose && println("We are at step ", i-1, "-> ",i)
//         if find_depth(T) != nothing
//             if OrbitRepresentation == minimal_image || length(T.L[i]) < 10000
//                 current_multi_threaded = false
//             else
//                 current_multi_threaded = multi_threaded
//             end
//             verbose && println("Multi threaded: ", current_multi_threaded)
//     		verbose && println("Nodes in current level:"*string(length(T.L[i])))
//     		verbose && println("Nodes total:"*string(sum([length(((T.L[j]))) for j in 1:length(T.L)])))
//             time_data = @timed delete_restrict!(T, OrbitRepresentation, current_multi_threaded)
//             verbose && println(time_data.time, " seconds")
//             verbose && print("\n")
// 			verbose && println(sum(sum(T.Betti)))
//             flush(stdout)
//         end
//     end
// 	verbose && println("Finished")
//     return nothing
// end

// ###############################
// #### Factors of main algorithm
// ###############################

// #This function scans through each key in a particular level and calls `branch`
// #  on those nodes
// function delete_restrict!(T::Hyperplane_Tree, OrbitRepresentation, multi_threaded)
// 	depth = find_depth(T)
// 	if depth == nothing
// 		println("Already finished")
// 		return nothing
// 	end

// 	#Extract size of problem
//     m, n = size(T.Hyperplanes[1])

//     AllLeaves = T.L
// 	sizehint!(AllLeaves[depth+1],  2*length(AllLeaves[depth]))

//     if multi_threaded
//     	Threads.@threads for CurrentRestriction in collect(keys(AllLeaves[depth]))
//     		branch!(T, CurrentRestriction, depth, m, n, OrbitRepresentation)
//         end
//     else
//         for CurrentRestriction in keys(AllLeaves[depth])
//             branch!(T, CurrentRestriction, depth, m, n, OrbitRepresentation)
//         end
//     end

// 	T.L[depth] = Dict{T.hash_type, Node{T.hash_type}}()

// 	return nothing
// end

// function branch!(T::Hyperplane_Tree, CurrentHashedKey, depth::Int64, m::Int64, n::Int64, OrbitRepresentation)

// 	tid = Threads.threadid()

// 	CurrentNode = T.L[depth][CurrentHashedKey]
//     multiplicity = CurrentNode.multiplicity
//     k = revert_hash!(T.TmpRestArray[tid],CurrentNode.hashed_rest,T.hash_base)
//     if T.Hyperplanes[tid] isa Matrix
//     	T.TmpHyperplanes[tid] .= T.Hyperplanes[tid]
//     else
// 		map!(identity, T.TmpHyperplanes[tid], T.Hyperplanes[tid])
// 	end
	

// 	#First restrict with respect to the old indices.
// 	#This gives the left_next_unbroken index.
// 	restrict_hyperplanes!(T.TmpHyperplanes[tid], T.TmpRestArray[tid], k, depth)
// 	left_next_unbroken = next_non_broken_elt(T, view(T.TmpHyperplanes[tid],k+1:m,depth+1:n), m)+depth

// 	#Second, perform another pivot step on column depth for the right next unbroken index.
// 	pivot_step!(T.TmpHyperplanes[tid], depth, k+1, depth, m, n)
//     right_next_unbroken = next_non_broken_elt(T, view(T.TmpHyperplanes[tid],k+2:m,depth+1:n), m)+depth

//     #####Right child construction

// 	#if the next unbroken is at the penultimate step
// 	#  we know it will contribute its multiplicity
// 	#  to betti_k+1 & betti_k+2
// 	if right_next_unbroken == n
// 	    @inbounds T.Betti[tid][k+2] += multiplicity
// 	    @inbounds T.Betti[tid][k+3] += multiplicity
// 	elseif right_next_unbroken == n-1
//         @inbounds T.Betti[tid][k+2] += multiplicity
//         @inbounds T.Betti[tid][k+3] += multiplicity*2
//         @inbounds T.Betti[tid][k+4] += multiplicity
//     else
// 		place_right_leaf!(T, CurrentNode, k, depth, right_next_unbroken, OrbitRepresentation)
// 	end

// 	#####Left child construction
// 	if depth == n-1
//         @inbounds T.Betti[tid][k+1] += multiplicity
//         @inbounds T.Betti[tid][k+2] += multiplicity
// 	else
// 		#If k==0 then we already know that the first hyperplane will be the next unbroken
// 		#  and since this happens once at every level, it's worth the extra code block
// 		if k == 0
//             place_left_leaf!(T, CurrentNode, k, depth, depth+1, OrbitRepresentation)
//         else
// 			if left_next_unbroken == n
//                 @inbounds T.Betti[tid][k+1] += multiplicity
//                 @inbounds T.Betti[tid][k+2] += multiplicity
// 			elseif left_next_unbroken == n-1
//                 @inbounds T.Betti[tid][k+1] += multiplicity
//                 @inbounds T.Betti[tid][k+2] += multiplicity*2
//                 @inbounds T.Betti[tid][k+3] += multiplicity
//             else
// 				place_left_leaf!(T, CurrentNode, k, depth, left_next_unbroken, OrbitRepresentation)
// 			end
// 		end
// 	end
// 	return nothing
// end

// function place_right_leaf!(T::Hyperplane_Tree, CurrentNode::Node, k, depth::Int64, target_depth::Int64, OrbitRepresentation)

//     tid = Threads.threadid()
//     @inbounds T.TmpRestArray[tid][k+1] = depth
//     rest = view(T.TmpRestArray[tid], 1:k+1)

// 	if OrbitRepresentation == trivial_minimal_image

// 		FN = Node{T.hash_type}(key_hash(rest,T.hash_type, T.hash_base), deepcopy(CurrentNode.multiplicity))
//         begin
//             lock(T.DictLockArray[target_depth])
//             try
//                 T.L[target_depth][FN.hashed_rest] = FN
//             finally
//                 unlock(T.DictLockArray[target_depth])
//             end
//         end
// 		return nothing
// 	end

//     Gens = T.GensDict[tid][target_depth]
//     Stab = T.StabilizerDict[target_depth]
//     MinimalImage = OrbitRepresentation(T, Gens, Stab, rest)
//     MinimalImageHash = key_hash(MinimalImage,T.hash_type,T.hash_base)
//     #Check if this minimal element has already been seen
//     begin
//         lock(T.DictLockArray[target_depth])
//         try
//             MinimalNode = get(T.L[target_depth], MinimalImageHash, nothing)
//             if MinimalNode != nothing
//                 #if so, simply increment where it's been seen by the multiplicity
//                 MinimalNode.multiplicity += CurrentNode.multiplicity
//             else
//                 #otherwise make a new node, place it at the next level which could
//                 # have a right node, record that we've seen the minimal image before
//                 # and set the nodes value in the leaf dictionary equal to its mult.
//                 FN = Node{T.hash_type}(key_hash(rest,T.hash_type,T.hash_base), deepcopy(CurrentNode.multiplicity))
//                 T.L[target_depth][MinimalImageHash] = FN
//             end
//         finally
//             unlock(T.DictLockArray[target_depth])
//         end
//     end

// 	#Finally, undo adding the depth as a restriction because we will use currentnode
// 	# for the left child
// 	T.TmpRestArray[tid][k+1] = 0
// end

// function place_left_leaf!(T::Hyperplane_Tree, CurrentNode::Node, k, depth, target_depth, OrbitRepresentation)

// 	tid = Threads.threadid()
//     rest = view(T.TmpRestArray[tid], 1:k)

// 	if OrbitRepresentation == trivial_minimal_image
//         begin
//             lock(T.DictLockArray[target_depth])
//             try
//                 T.L[target_depth][CurrentNode.hashed_rest] = CurrentNode
//             finally
//                 unlock(T.DictLockArray[target_depth])
//             end
//         end
// 		return nothing
// 	end

//     Gens = T.GensDict[tid][target_depth]
//     Stab = T.StabilizerDict[target_depth]

//     MinimalImage = OrbitRepresentation(T, Gens, Stab, rest)
//     MinimalImageHash = key_hash(MinimalImage,T.hash_type,T.hash_base)
//     begin
//         lock(T.DictLockArray[target_depth])
//         try
//             MinimalNode = get(T.L[target_depth], MinimalImageHash, nothing)
//             if MinimalNode != nothing
//                 #if so, simply increment where it's been seen by the multiplicity
//                 MinimalNode.multiplicity += CurrentNode.multiplicity
//                 CurrentNode = nothing
//             else
//                 T.L[target_depth][MinimalImageHash] = CurrentNode
//             end
//         finally
//             unlock(T.DictLockArray[target_depth])
//         end
//     end
// end


// #####################################################
// ##### Linear algebra
// #####################################################

// function coned_arrangement(H::Union{Array{T,2},MatElem}, ConstantTerms::Vector{T}) where {T <: Union{Nemo.RingElem,Integer}}
//     @assert size(H,2) == length(ConstantTerms)
//     #Add constant terms as new row, i.e. as homogenized with a new variable
//     ConeArr = vcat(H, ConstantTerms')
//     #Add hyperplane x_{d+1}=0
//     HyperplaneAtInfinity = zeros(parent(H[1,1]), size(ConeArr, 1))
//     HyperplaneAtInfinity[end,1] = parent(H[1,1])(1)
//     ConeArr = hcat(ConeArr, HyperplaneAtInfinity)
// end


// #Perform Pivot steps on the matrix M on the column indices in restrictions
// #The changes in M are only carried out for columns with index above depth
// function restrict_hyperplanes!(M, rest::Array{Int64,1}, k, depth)
//     nr, nc = size(M)
//     for r in 1:k
// 	    # find first nonzero entry in the first column
// 	    pivot_step!(M, rest[r], r, depth, nr, nc; extra_indices=view(rest,r+1:k))
// 	end

//     return nothing
// end

// function pivot_step!(M, col_index, firstr, depth, nr, nc; extra_indices=nothing)

// 	firstnz = firstr

//     while firstnz <= size(M)[1] && iszero(M[firstnz,col_index])
//         firstnz += 1
//     end
//     if firstnz > nr
//     	return
//     end

//     if col_index == depth
//     	start_tail_index = depth+1
//     else
//     	start_tail_index = depth
//     end

//     # Swap firstr and firstnz row
//     if firstnz > firstr
//     	M[firstr,col_index], M[firstnz,col_index] = M[firstnz, col_index], M[firstr,col_index]
//     	if extra_indices != nothing
// 	        @simd for k in extra_indices
// 	            M[firstr,k], M[firstnz,k] = M[firstnz, k], M[firstr,k]
// 	        end
// 	    end

//         @simd for k=start_tail_index:nc
//             M[firstr, k], M[firstnz, k] = M[firstnz, k], M[firstr, k]
//         end
//     end

//     d = M[firstr,col_index]
//     d_isone = (d == 1)

//     for i=firstr+1:nr
//         a = (-1)*M[i,col_index]
//         if !iszero(a)
// 	        if extra_indices != nothing
// 		        @simd for k in extra_indices
// 		            if d_isone
// 		                M[i,k] += M[firstr,k]*a
// 		            else
// 		                M[i,k] = M[firstr,k]*a+M[i,k]*d
// 		            end
// 		        end
// 	    	end
// 	        @simd for k=start_tail_index:nc
// 	            if d_isone
// 	                M[i,k] += M[firstr,k]*a
// 	            else
// 	                M[i,k] = M[firstr,k]*a+M[i,k]*d
// 	            end
// 	        end
// 	    end
//     end
// end

// # We want to find the first column of A that does not have a multiple column in the following matrix.
// function next_non_broken_elt(T, A, original_m::Int64)
//     m,n = size(A)
//     if m == original_m
//         # Nothing can be broken at this stage. So we just return the next element
//         return 1
//     elseif m == 1
//         # Only one row left. Thus, we can return the last column of the original matrix.
//         return n
//     else
//         equal_columns = true
//         for current_col_ind in 1:n
//             equal_columns = false
//             compare_col_ind = current_col_ind+1
//             while compare_col_ind <= n && equal_columns == false
//                 equal_columns = compare_columns(A, current_col_ind, compare_col_ind,
//                     T.nemo_tmp1[Threads.threadid()], T.nemo_tmp2[Threads.threadid()])
//                 compare_col_ind += 1
//             end
//             if equal_columns == false
//                 return current_col_ind
//             end
//         end
//         return n+1
//     end
// end


// # True when the two columns are linearly dependent
// # The following code has been provided by Tommy Hoffmann
// function compare_columns(A::AbstractArray{T,2}, ind_v::Int64, ind_w::Int64, t1, t2)::Bool where {T<:Nemo.RingElem}
//     len = size(A,1)

//     firstnz = 1
//     GC.@preserve A begin
//         while firstnz <= len && iszero(A[firstnz,ind_v])
//             firstnz += 1
//         end

//         for i in 1:(firstnz-1)
//             if !iszero(A[i,ind_w])
//                 return false
//             end
//         end

//         if iszero(A[firstnz,ind_w])
//             return false
//         end

//         a = A[firstnz,ind_w]
//         b = A[firstnz,ind_v]

//         firstnz += 1
//         while firstnz <= len
//         	# This isn't working for some reason I don't understand.
//             #Nemo.mul!(t1, a, A[firstnz,ind_v])
//             #Nemo.mul!(t2, b, A[firstnz,ind_w])
//             t1 = a*A[firstnz,ind_v]
//             t2 = b*A[firstnz,ind_w]
//             if !isequal(t1, t2)
//                 return false
//             end
//             firstnz += 1
//         end
//   end
//   return true
// end


// # True when the two columns are linearly dependent
// # The following code has been provided by Tommy Hoffmann
// function compare_columns(A::Union{AbstractArray{T,2},MatElem}, ind_v::Int64, ind_w::Int64, t1, t2)::Bool where {T <: Integer}
//     len = size(A,1)

//     firstnz = 1

//     while firstnz <= len && iszero(A[firstnz,ind_v])
//         firstnz += 1
//     end
//     #@assert firstnz <= len
//     if firstnz > len 
//     	return true
//     end

//     for i in 1:(firstnz-1)
//         if !iszero(A[i,ind_w])
//             return false
//         end
//     end

//     if iszero(A[firstnz,ind_w])
//         return false
//     end

//     a = A[firstnz,ind_w]
//     b = A[firstnz,ind_v]

//     firstnz += 1
//     while firstnz <= len
//         if !isequal(a* A[firstnz,ind_v], b* A[firstnz,ind_w])
//             return false
//         end
//         firstnz += 1
//     end

//     return true
// end

// ######################################################
// #Group theory functions
// ######################################################


// function get_stabilizer(depth, n, G)
// 	if G==nothing
// 		return nothing
// 	end
// 	if depth == n
// 		throw(ArgumentError("Why are we computing at step n?"))
// 	end
// 	H = GAP.Globals.Stabilizer(G, GapObj(depth:n), GAP.Globals.OnSets)
// end

// function get_stoch_gens(H,n, proportion, max_size, min_size)
// 	size_H = GAP.Globals.Size(H)
// 	if size_H > 10000 # Otherwise prop can't be computed...
// 		Gens = produce_many_permutations(H, n, min_size)
// 	else
// 		prop = Int64(ceil(size_H*proportion))
// 		actual_number_of_gens = Int64(min(max(min_size, prop), max_size))
// 		if size_H <= actual_number_of_gens
// 			Gens = produce_all_permutations(H, n, size_H)
// 		else
// 			Gens = produce_many_permutations(H, n, actual_number_of_gens)
// 		end
// 	end
// end

// function minimal_image(T, Gens::Array{Array{Int64,1},1}, G::GAP.GapObj, I::AbstractArray{Int64,1})::Array{Int64,1}
// 	#last two arguments do nothing; they're there only to be consistent with stochastic_greedy
//     m=GAP.Globals.MinimalImage(G,GapObj(Array(I)),GAP.Globals.OnSets)
//     K=GAP.gap_to_julia(m)
//     return(K)
// end

// function canonical_image(T, Gens::Array{Array{Int64,1},1}, G::GAP.GapObj, I::AbstractArray{Int64,1})::Array{Int64,1}
// 	#last two arguments do nothing; they're there only to be consistent with stochastic_greedy
//     m=GAP.Globals.CanonicalImage(G,GapObj(Array(I)),GAP.Globals.OnSets)
//     K=GAP.gap_to_julia(m)
//     return(K)
// end

// #returns a permutation from gap to julia
// function gap_to_julia_perm(g::GAP.GapObj, N::Int64)::Array{Int64,1}
//     GAP.gap_to_julia(GAP.Globals.ListPerm(g,N))
// end

// #If G is very small, i.e. at most 20 we return the entire group assuming k=|G|
// function produce_all_permutations(G::GAP.GapObj ,N::Int64, k::Int64)::Array{Array{Int64,1},1}
//     elements=GAP.Globals.List(G)
//     perms = []
//     for i in 2:k
//         perms = push!(perms,gap_to_julia_perm(elements[i],N))
//     end
//     return(perms)
// end

// function produce_many_permutations(G::GAP.GapObj ,N::Int64, k::Int64)::Array{Array{Int64,1},1}
//     Ggens = GAP.Globals.GeneratorsOfGroup(G)
//     ngens = length(Ggens)
//     JuliaGens = [gap_to_julia_perm(Ggens[i],N) for i in 1:ngens]
//     perms = JuliaGens[[1]]
//     counter = 0
//     while counter<1000 && length(perms) < k
//         g = gap_to_julia_perm(GAP.Globals.PseudoRandom(G), N)
//         if (g in perms) == false
//             perms = push!(perms, g)
//         else
//             counter += 1
//         end
//     end
//     return(perms)
// end

// function greedy_min_elt(I::AbstractArray{Int64,1}, J::AbstractArray{Int64,1}, Gens::Array{Array{Int64,1},1})

//     j_is_smaller = false::Bool
//     size_I = length(I)

//     #apply each g in Gens to I, if anything becomes smaller, call the function again
//     for g_i in eachindex(Gens)
//         j_is_smaller = false
//         i = 1::Int64
//         @simd for i=1:size_I
//             @inbounds J[i] = Gens[g_i][I[i]]
//         end
//         sort!(J)

//         if J < I
//             return greedy_min_elt(J, I, Gens)
//         end
//     end
//     return I
// end


// function pseudo_minimal_image(T, Gens::Array{Array{Int64,1},1}, G::GAP.GapObj, I::AbstractArray{Int64,1})::Array{Int64,1}
// 	I_new = T.I_array_cache[Threads.threadid()][length(I)]
//     J_new = T.J_array_cache[Threads.threadid()][length(I)]
//     @simd for i in eachindex(I)
//     	@inbounds I_new[i] = I[i]
//     end
//     return greedy_min_elt(I_new,J_new,Gens)
// end

// function trivial_minimal_image(T, Gens::Array{Array{Int64,1},1}, G::GAP.GapObj, I::AbstractArray{Int64,1})::Array{Int64,1}
//     return I
// end

// ###########################
// #### Helper Functions
// ###########################

// #Collision free hash function for Arrays (up to threshold 9)
// #Save these as minimal_image keys instead of Arrays
// #Save an Array as sum i*2^b*i-1, where b is a hash_base which should be bigger than log_2(n)
// # where n is the number of hyperplanes.
// function key_hash(I,hash_type,hash_base::Int64)
// 	result = hash_type(0)
// 	@simd for i in eachindex(I)
// 		@inbounds result += I[i]*hash_type(2)^(hash_base*(i-1))
// 	end
// 	@assert result >= 0
// 	return result
// end


// # Reverts the above hash function.
// # Saves the array in I and returns its length
// function revert_hash!(I::Array{Int64,1},key,hash_base::Int64)
// 	i = 0
// 	while key > 0
// 		i += 1
// 		@inbounds I[i] = key % 2^hash_base
// 		key ÷= 2^hash_base
// 	end
// 	@simd for j=i+1:length(I)
// 		@inbounds I[j]=0
// 	end
// 	return i
// end


// function find_depth(T::Hyperplane_Tree)
// 	loc = findfirst(x->length(keys(x))>0,T.L)
// 	return loc
// end

// end
