immutable SparseRegPath{L <: Loss, P <: Penalty}
    loss::L
    penalty::P
    factor::VecF
end
