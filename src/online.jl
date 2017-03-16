immutable StreamReg{
            A <: OnlineAlgorithm,
            L <: Loss,
            P <: Penalty,
            W <: Weight
        } <: AbstractSparseReg
    β::VecF
    loss::L
    penalty::P
    algorithm::A
    λ::Float64
    factor::VecF
    weight::W
    η::Float64
end
