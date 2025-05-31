function vander(a::Vector{Float64}, d::Int)

    n = length(a)
    V = zeros(n, d+1)

    for i = 1:n
        for j = 1:d+1
            V[i, j] = a[i]^(j-1)
        end
    end

    return V
end

function local_fdr(zscores; null_proportion=1.0, null_pdf=nothing, deg=7, nbins=30)

    # Calculate local FDR values for a list of Z-scores.

    # Parameters
    # ----------
    # zscores : array_like
    #     A vector of Z-scores
    # null_proportion : float
    #     The assumed proportion of true null hypotheses
    # null_pdf : function mapping reals to positive reals
    #     The density of null Z-scores; if None, use standard normal
    # deg : int
    #     The maximum exponent in the polynomial expansion of the
    #     density of non-null Z-scores
    # nbins : int
    #     The number of bins for estimating the marginal density
    #     of Z-scores.

    # Returns
    # -------
    # fdr : array_like
    #     A vector of FDR values

    # References
    # ----------
    # B Efron (2008).  Microarrays, Empirical Bayes, and the Two-Groups
    # Model.  Statistical Science 23:1, 1-22.

    # Examples
    # --------
    # Basic use (the null Z-scores are taken to be standard normal):

    # >>> from statsmodels.stats.multitest import local_fdr
    # >>> import numpy as np
    # >>> zscores = np.random.randn(30)
    # >>> fdr = local_fdr(zscores)

    # Use a Gaussian null distribution estimated from the data:

    # >>> null = EmpiricalNull(zscores)
    # >>> fdr = local_fdr(zscores, null_pdf=null.pdf)

    # Bins for Poisson modeling of the marginal Z-score density
    minz, maxz = extrema(zscores)
    maxz += (maxz - minz) / 10000
    zhist = zeros(nbins)
    f = (maxz - minz) / nbins
    for i in eachindex(zscores)
        j = Int(floor((zscores[i] - minz) / f)) + 1
        zhist[j] += 1
    end

    # Bin centers
    zbins = zeros(nbins)
    for i in 1:nbins
        zbins[i] = minz + (i - 0.5) * f
    end

    # The design matrix at bin centers
    dmat = vander(zbins, deg)

    # Poisson regression
    md = fit(GeneralizedLinearModel, dmat, zhist, Poisson())

    # The design matrix for all Z-scores
    dmat_full = vander(zscores, deg)

    # The height of the estimated marginal density of Z-scores,
    # evaluated at every observed Z-score.
    fz = exp.(dmat_full * coef(md)) / (length(zscores) * f)

    # The null density.
    if null_pdf == nothing
        f0 = exp.(-0.5 .* zscores .^ 2) ./ sqrt(2 * pi)
    else
        f0 = null_pdf(zscores)
    end

    # The local FDR values
    fdr = null_proportion * f0 ./ fz

    clamp!(fdr, 0, 1)

    return fdr
end
