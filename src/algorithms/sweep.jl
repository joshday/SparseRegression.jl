immutable SWEEP <: OfflineAlgorithm
    S::Matrix{Float64}
end
SWEEP(p::Integer = 0) = SWEEP(zeros(p + 1, p + 1))


is_supported(loss::LinearRegression, pen::Union{NoPenalty, L2Penalty}, alg::SWEEP) = true
