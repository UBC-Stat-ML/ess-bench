using AdvancedMH
using Distributions
using MCMCChains
using DataFrames 

struct TuringESS end 

struct InformedESS{T}
    target::T
end

function run_mh(proposal_sd, target, n_iterations) 
    target_logd(x) = logpdf(target, x)
    model = DensityModel(target_logd)
    sampler = RWMH(Normal(0, proposal_sd))
    return sample(model, sampler, n_iterations, chain_type=Chains; progress=false)
end

compute_ess(samples, ::TuringESS) = ess(samples)

function compute_ess(samples, informed::InformedESS)
    target = informed.target
    
    posterior_mean = mean(target)
    posterior_sd = std(target)
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
    
    df = DataFrame(seed = Int[], type = Symbol[], moment = Int[], value = Float64[])
    for seed in 1:n_repeats
        c = run_mh(proposal_sd, Normal(0, 1) , n_iterations)
        samples = c[:param_1]
        for moment in [1, 2]
            ref = moment == 1 ? Normal(0, 1) : Chi(1)
            for ess_type in [TuringESS(), InformedESS(ref)]
                push!(df, (;
                    seed, 
                    moment,
                    type = Symbol(Base.typename(typeof(ess_type)).name), 
                    value = compute_ess(samples .^ moment, ess_type)
                ))
            end
        end
    end

    return df
end

main(1.0, 1, 100)