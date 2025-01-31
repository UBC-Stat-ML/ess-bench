using Distributions
using MCMCChains
using DataFrames 

struct TuringESS{K}
    args::K 
end 
descr(t::TuringESS) = "TuringESS$(t.args)"

struct BatchMeanESS{T}
    target::T
end
descr(::BatchMeanESS{Nothing}) = "BatchMeanESS"
descr(::BatchMeanESS) = "BatchMeanESS(informed)"

ess_estimators(ref) = [
    BatchMeanESS(ref), 
    BatchMeanESS(nothing),
    TuringESS((; kind = :bulk)),
    TuringESS((; kind = :basic)),
]

function run_mh(proposal_sd, target, n_iterations) 
    target_logd(x) = logpdf(target, x)
    result = zeros(n_iterations) 
    current_point = randn()
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

compute_ess(samples, t::TuringESS) = ess(samples; t.args...)

function compute_ess(samples, informed::BatchMeanESS)
    target = informed.target
    if target === nothing
        posterior_mean = mean(samples) 
        posterior_sd = std(samples)
    else
        posterior_mean = mean(target)
        posterior_sd = std(target)
    end
    n_samples = length(samples)
    n_blocks = 1 + isqrt(n_samples)
    blk_size = n_samples ÷ n_blocks # takes floor of division
    centered_batch_means = map(1:n_blocks) do b
        i_start = blk_size*(b-1) + 1
        i_end = blk_size*b
        mean(x -> (x - posterior_mean)/posterior_sd, @view samples[i_start:i_end])
    end
    n_blocks / mean(abs2, centered_batch_means)
end

function main(proposal_sd, n_repeats, n_iterations)
    
    df = DataFrame(seed = Int[], type = String[], moment = Int[], value = Float64[])
    for seed in 1:n_repeats
        samples = run_mh(proposal_sd, Normal(0, 1) , n_iterations)
        for moment in [1, 2]
            ref = moment == 1 ? Normal(0, 1) : Chisq(1)
            for ess_type in ess_estimators(ref)
                push!(df, (;
                    seed, 
                    moment,
                    type = descr(ess_type), 
                    value = compute_ess(samples .^ moment, ess_type)
                ))
            end
        end
    end

    return df
end

main(1.0, 1, 100000)
