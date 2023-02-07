using triangular-lattice
using Documenter

DocMeta.setdocmeta!(triangular-lattice, :DocTestSetup, :(using triangular-lattice); recursive=true)

makedocs(;
    modules=[triangular-lattice],
    authors="Wei Tang <tangwei@smail.nju.edu.cn> and contributors",
    repo="https://github.com/tangwei94/triangular-lattice.jl/blob/{commit}{path}#{line}",
    sitename="triangular-lattice.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tangwei94.github.io/triangular-lattice.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tangwei94/triangular-lattice.jl",
)
