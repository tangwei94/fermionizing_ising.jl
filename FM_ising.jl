using TensorKit, MPSKit

A = TensorMap(rand, ComplexF64, ℂ^6, ℂ^6)
ψ = InfiniteMPS([A])