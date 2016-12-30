immutable SWEEP <: OfflineAlgorithm
    S::Matrix{Float64}
end
SWEEP() = SWEEP(zeros(0, 0))
init(alg::SWEEP, p::Integer) = SWEEP(zeros(p + 1, p + 1))



is_supported(loss::LinearRegression, pen::Union{NoPenalty, L2Penalty}, alg::SWEEP) = true
