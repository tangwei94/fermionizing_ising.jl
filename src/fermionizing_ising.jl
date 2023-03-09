__precompile__()

module fermionizing_ising

    using LinearAlgebra
    using Polynomials

    export kramers, βc_honeycomb, f_density_honeycomb_α, f_density_honeycomb
    export βc_triangular, f_density_triangular

    include("honeycomb.jl");
    include("triangular.jl");

end
