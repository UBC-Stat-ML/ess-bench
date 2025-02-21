using Distributions
using MCMCChains
using DataFrames 
using Random 

struct TuringESS{K}
    maxlag::K
    fft::Bool 
end 
descr(t::TuringESS) = "maxlag = $(t.maxlag), fft=$(t.fft)"
descr(t::TuringESS{Nothing}) = "maxlag = default(250), fft=$(t.fft)"
family(::TuringESS) = "Spectral"

struct BatchMeanESS{T}
    target::T
    rate::Float64 
end
descr(bm::BatchMeanESS{Nothing}) = "rate = $(bm.rate), uninf"
descr(bm::BatchMeanESS) = "rate = $(bm.rate), inform"
family(::BatchMeanESS) = "BatchMean"

ess_estimators(ref) = [
    BatchMeanESS(ref, 0.3), 
    BatchMeanESS(nothing, 0.3),
    BatchMeanESS(ref, 0.5), 
    BatchMeanESS(nothing, 0.5),
    BatchMeanESS(ref, 0.7), 
    BatchMeanESS(nothing, 0.7),
    TuringESS(nothing, true),
    TuringESS(Inf, true),
    TuringESS(nothing, false),
    TuringESS(Inf, false),
]

function run_mh(proposal_sd, target, n_iterations, initialization) 
    target_logd(x) = logpdf(target, x)
    result = zeros(n_iterations) 
    current_point = rand(initialization)
    for i in 1:n_iterations 
        proposed_point = current_point + proposal_sd * randn() 
        log_ratio = target_logd(proposed_point) - target_logd(current_point) 
        if rand() < exp(log_ratio) 
            current_point = proposed_point 
        end 
        result[i] = current_point 
    end
    return result
end

compute_ess(samples, t::TuringESS) = ess(samples; autocov_method = t.fft ? FFTAutocovMethod() : AutocovMethod(), maxlag_arg(t.maxlag)...)
maxlag_arg(::Nothing) = (;)
maxlag_arg(maxlag) = (; maxlag = isinf(maxlag) ? typemax(Int) : maxlag)

function compute_ess(samples, bm::BatchMeanESS)
    target = bm.target
    posterior_mean, posterior_sd = isnothing(target) ? (mean(samples), std(samples)) : (mean(target), std(target))
    n_samples = length(samples)
    n_blocks = 1 + floor(Int, n_samples^(1.0 - bm.rate)) # isqrt(n_samples)
    blk_size = n_samples ÷ n_blocks # takes floor of division
    centered_batch_means = map(1:n_blocks) do b
        i_start = blk_size*(b-1) + 1
        i_end = blk_size*b
        mean(x -> (x - posterior_mean)/posterior_sd, @view samples[i_start:i_end])
    end
    n_blocks / mean(abs2, centered_batch_means)
end

function main(proposal_sd, n_repeats, n_iterations)
    #Random.seed!(1)
    df = DataFrame(seed = Int[], type = String[], family = String[], moment = Int[], ess = Float64[], time = Float64[], initialization = String[])
    for seed in 1:n_repeats
        for init in [Normal(0, 0.1), Normal(0, 1), Normal(0, 5), Normal(5, 1)]
            samples = run_mh(proposal_sd, Normal(0, 1) , n_iterations, init)
            for moment in [2]
                ref = moment == 1 ? Normal(0, 1) : Chisq(1)
                for ess_type in ess_estimators(ref)
                    compute_ess(samples .^ moment, ess_type) # warm up, for timing purpose
                    tuple = @timed compute_ess(samples .^ moment, ess_type)
                    @show tuple
                    @show tuple.value
                    push!(df, (;
                        seed, 
                        moment,
                        initialization = string(init),
                        type = descr(ess_type), 
                        family = family(ess_type),
                        ess = tuple.value,
                        time = tuple.time
                    ))
                end
            end
        end
    end

    return df
end

# main(1.0, 1, 100000)
