function calculateLogProb(x,p::PrefEl)
	out = 0.0

	# println("x = $x")

	for i in 1:size(p.strict,1)
		out += getLogStrictProb(x,p.data[p.strict[i,1],:],p.data[p.strict[i,2],:],p.Sigma)
	end

	for i in 1:size(p.indif,1)
		out += getLogIndifProb(x,p.data[p.indif[i,1],:],p.data[p.indif[i,2],:],p.Sigma)
	end

	out += logPrior(x,p.priors.dists)

	println(out)

	return out 
end

function getLogStrictProb{R<:Real}(x::Array{R,1}, a::Array{R,2}, b::Array{R,2}, Sigma::Array{R,2})
	d = a .- b
	varAlongD = d * Sigma * d'
	meanAlongD = dot(vec(d),vec(x))
	temp = Normal(0,sqrt(varAlongD[1]))
	return logcdf(temp,meanAlongD)
end

function getLogIndifProb{R<:Real}(x::Array{R,1}, a::Array{R,2}, b::Array{R,2}, Sigma::Array{R,2})
	d = a .- b
	varAlongD = (d * Sigma * d')[1]
	meanAlongD = dot(vec(d),x)
	temp = Normal(0,sqrt(varAlongD))
	return log(cdf(temp,meanAlongD + 0.5) - cdf(temp,meanAlongD - 0.5))
end

function logPrior{R <: Real}(x::Vector{R}, priors::Vector{Distribution})
	out = 0.0
	@simd for i in 1:length(x)
		if typeof(priors[i]) != Uniform || priors[i].a != -Inf
			@inbounds out += logpdf(priors[i],x[i])
		end
	end
	if out == -Inf
		out = realmin()
	end
	return out
end

using NLopt
function infer(p::PrefEl; method = "MAP")

	if method == "MAP"

		# Set up optimization
		n = size(p.data,2)
		opt = Opt(:LN_NEWUOA_BOUND,n) # slow
		# opt = Opt(:LN_BOBYQA, n) # really slow
		# opt = Opt(:LN_COBYLA, n) # slow
		# opt = Opt(:LD_LBFGS,n)
		xtol_rel!(opt,1e-4)
		# ftol_rel!(opt,1e-8)
		
		# Create dummy function
		f(x,grad) = calculateLogProb(x,p)

		# Want MAP estimate
		max_objective!(opt,f)

		# Restrict optimization to be within support of prior
		# necessary for priors like the Exponential dist
		lb = zeros(n)
		ub = zeros(n)
		for i in 1:n
			s = support(p.priors.dists[i])
			lb[i] = max(s.lb,p.priors.lb[i])
			ub[i] = min(s.ub,p.priors.ub[i])
		end
		lower_bounds!(opt,lb)
		upper_bounds!(opt,ub)

		# Set the starting point to be the prior mode
		means = zeros(F,n)
		for i in 1:n
			means[i] = mean(p.priors.dists[i]) # Would like to use mode, but exponential will kill autodiff because it'll check negative values
			if isnan(means[i])
				means[i] = 0.0
			end
		end

		minF, maxX, ret = optimize(opt,means)

		println(size(maxX))

		println("Optimization returned with code $ret")

		return maxX

	elseif method == "MCMC"
		samples = getMCMCSamples(p)
		return vec(mean(samples,1))
	else
		error("Unknown method $method")
	end
end

function suggest(p::PrefEl)
	n = size(p.data,1)

	# Set up all possible recommendations
	recs = cell(div(n * (n-1),2)) # to maintain int-ness
	count = 1
	for i in 1:n
		for j in i+1:n
			recs[count] = (i,j)
			count += 1
		end
	end

	# Create dummy function for pmap
	f(x) = estimateEntropy(p,x[1],x[2])

 	# Map f to our candidates
	results = pmap(f,recs)

	# Find the optimum
	_, id = findmin(results)

	return recs[id]
end

using KernelDensity
function estimateEntropy(p::PrefEl, a::Int,b::Int)
	pNew = deepcopy(p) # need pass-by-value, or else procs will talk over each other and die

	# Get the probability that a > b
	est = vec(infer(p))
	abProb = exp(getLogStrictProb(est,p.data[a,:],p.data[b,:],p.Sigma))

	# Get samples from posterior if a > b
	pNew.strict = [pNew.strict; a b]
	abSamples = getMCMCSamples(pNew)

	# Get samples from posterior if b > a
	pNew.strict = [pNew.strict[1:end-1,:]; b a]
	baSamples = getMCMCSamples(pNew)

	# Calculate entropies for each solution
	aEntropy = 0.0
	bEntropy = 0.0
	for dim in 1:size(abSamples,2)
		aKernel = KernelDensity.kde(abSamples[:,dim])
		bKernel = KernelDensity.kde(baSamples[:,dim])
		aEntropy += dot(aKernel.density[:], log(aKernel.density[:]))
		bEntropy += dot(bKernel.density[:], log(bKernel.density[:]))
	end	

	# Return expected entropy 
	return -(abProb * aEntropy + (1-abProb) * bEntropy)
end

function getMCMCSamples(p::PrefEl, samples = 1000)
	n = size(p.data,2)

	sigma = 1.0
	C = eye(n) * sigma
	propDist = MvNormal(C)

	samps = zeros(samples,n)

	samps[1,:] = (infer(p))'
	zOld = calculateLogProb(vec(samps[1,:]),p)

	for i in 2:samples
		prop = samps[i-1,:] .+ rand(propDist)'

		zNew = calculateLogProb(vec(samps[1,:]),p)

		if zNew > zOld || log(rand()) < zNew - zOld
			zOld = zNew
			samps[i,:] = prop
		else
			samps[i,:] = samps[i-1,:]
		end
	end

	return samps
end