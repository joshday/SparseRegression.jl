abstract SGDLike <: OnlineAlgorithm

type SGD <: SGDLike end
SGD(p::Int = 0) = SGD()
is_supported(loss::Loss, pen::Penalty, alg::SGDLike) = true
