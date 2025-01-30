include { instantiate; precompile; activate } from './nf-nest/pkg.nf'
include { crossProduct; filed; deliverables } from './nf-nest/cross.nf'
include { combine_csvs; } from './nf-nest/combine.nf'

def julia_env = file('julia_env')
def julia_script = file('code.jl')

def variables = [
    bandwidth: (-10..10).collect{ i -> Math.pow(2.0, i)}, 
]

workflow  {
    compiled_env = instantiate(julia_env) | precompile
    configs = crossProduct(variables)
    combined = run_julia(compiled_env, julia_script, configs) | combine_csvs
    plot(compiled_env, combined)

    // for each bandwidth, compute {ESS(mu, sigma), ESS} x {x, x^2}
    // if problem also occurs with x^2, device an alternative, else, think
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

    df = main(${config.bandwidth}, 10, 1_000)
    mkdir("${filed(config)}")
    CSV.write("${filed(config)}/ess.csv", df)
    """
}

process plot {
    debug true

    input:
        path julia_env 
        path combined_csvs_folder 

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
    transform!(df, :moment => ByRow(string) => :moment)

    plt = data(df) * mapping(:bandwidth, :value, color = :type, row = :moment) * visual(Scatter, alpha=0.25) 
    fg = draw(plt; 
            axis = (; xscale = log2, yscale = log2),
            figure = (; size = (500, 500))
        )
    save("ess.png", fg, px_per_unit = 3)
    """
}