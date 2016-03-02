module StatisticalLearningTests
using StatisticalLearning
using FactCheck, Distributions

macro title(s)
    return :("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ " * $s)
end
macro subtitle(s)
    return :("████████████████████████████████████████████████ " * $s)
end

include("testfiles/linpredmodel_test.jl")
end
