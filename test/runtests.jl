using LocalFDR
using Test
using StableRNGs
using Statistics

@testset "basic null" begin

    rng = StableRNG(123)
    n = 200
    nrep = 100
    for k in 1:nrep
        x = randn(rng, n)
        f = local_fdr(x)
        @test sum(f .<= 0.1) <= 2
    end
end

@testset "basic spiked" begin

    rng = StableRNG(123)
    n = 200
    nrep = 100
    efdr = zeros(nrep)
    power = zeros(nrep)
    for k in 1:nrep
        x = randn(rng, n)
        x[1:10] .*= 10
        f = local_fdr(x; deg=5)
        efdr[k] = sum(f[11:end] .<= 0.1) / sum(f .<= 0.1)
        power[k] = mean(f[1:10] .<= 0.1)
    end
    @test mean(efdr) < 0.1
    @test mean(power) > 0.5
end
