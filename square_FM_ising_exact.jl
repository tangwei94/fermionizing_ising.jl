function f_density_square(N::Int, β::Real)

    β1 = -0.5 * log(tanh(β)) # β^\ast

    M1 = zeros(ComplexF64, 2*N, 2*N)
    M2 = zeros(ComplexF64, 2*N, 2*N)

    for ix in 1:N
        M1[2*ix-1, 2*ix] = 2*β1
        M1[2*ix, 2*ix-1] = -2*β1
    end

    # TODO. parity!
    for ix in 1:N
        M2[2*ix, (2*ix+1) % (2*N)] = 2*β
        M2[(2*ix+1) % (2*N), 2*ix] = -2*β
    end

    RT = exp(-im * M1 / 2) * exp(-im * M2) * exp(-im * M1 / 2)

    Λ, _ = eigen(Hermitian(RT))

    ϵs = log.(Λ)[N+1:end]

    f = -sum(ϵs) / (2*N*β) - log(2*sinh(2*β)) / (2*β)
    return f
end

βc = asinh(1) / 2
fs = free_energy_density_latt.(8:8:98, βc)

N = 100 

Δβ = 0.001
βs = Vector(βc .+ Δβ .* (-20:20))
fs = f_density_square.(N, βs)
Es = (fs[3:end] - fs[1:end-2]) / (2*Δβ) .* βs[2:end-1]
Cvs = -(Es[3:end] - Es[1:end-2]) / (2*Δβ) .* (βs[3:end-2] .^ 2)
#itp = interpolate((βs,), fs, Gridded(Linear()));
#Es = only.(Interpolations.gradient.(Ref(itp), βs)) .* βs
#itp2 = interpolate((βs,), Es, Gridded(Linear()));
#Cvs = only.(Interpolations.gradient.(Ref(itp2), βs)) .* (-βs .^ 2)

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 800))

ax1 = Axis(fig[1, 1], 
        xlabel = L"\beta",
        ylabel = L"f", 
        )

sc1_orig = scatter!(ax1, βs, fs, marker=:o, markersize=10)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"\beta",
        ylabel = L"E", 
        )

sc2_orig = scatter!(ax2, βs[2:end-1], Es, marker=:o, markersize=10)
@show fig

ax3 = Axis(fig[3, 1], 
        xlabel = L"\beta",
        ylabel = L"C_V", 
        )

sc3_orig = scatter!(ax3, βs[3:end-2], Cvs, marker=:o, markersize=10)
@show fig

save("square_lattice.pdf", fig)