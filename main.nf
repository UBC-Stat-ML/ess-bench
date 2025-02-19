include { instantiate; precompile; activate } from './nf-nest/pkg.nf'
include { crossProduct; filed; deliverables } from './nf-nest/cross.nf'
include { combine_csvs; } from './nf-nest/combine.nf'

def julia_env = file('julia_env')
def julia_script = file('code.jl')


def variables = [
    bandwidth: (-20..20).collect{ i -> Math.pow(2.0, i)}, 
    n_samples: [100_000]
]

workflow  {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables)
    combined = run_julia(compiled_env, julia_script, configs) | combine_csvs
    plot(compiled_env, combined, ['ess', 'time'])
}

process run_julia {
    input:
        path julia_env 
        path julia_script
        val config 
    output:
        path "${filed(config)}"
    """
    ${activate(julia_env)}

    # run your code
    include("$julia_script")
    using CSV 

    df = main(${config.bandwidth}, 10, ${config.n_samples})
    mkdir("${filed(config)}")
    CSV.write("${filed(config)}/ess.csv", df)
    """
}

process plot {
    debug true

    input:
        path julia_env 
        path combined_csvs_folder 
        each output

    output:
        path combined_csvs_folder
        path '*.png'
    publishDir "${deliverables(workflow, params)}", mode: 'copy', overwrite: true

    """
    ${activate(julia_env)}

    using CSV 
    using DataFrames
    using AlgebraOfGraphics
    using CairoMakie

    df = CSV.read("$combined_csvs_folder/ess.csv", DataFrame)
    n_samples = df[1, :n_samples]

    plt = data(df) * mapping(:bandwidth, :$output, col = :type, color = :family, row = :initialization) * visual(Scatter, alpha=0.25) 
    
    fg = draw(plt ${output == "ess" ? "+ mapping([n_samples, sqrt(n_samples)]) * visual(HLines)" : ""}; 
            axis = (; xscale = log2, yscale = log2),
            figure = (; size = (2400, 1000))
        )
    save("${output}.png", fg, px_per_unit = 3)
    """
}