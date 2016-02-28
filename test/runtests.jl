module StatisticalLearningTests
using StatisticalLearning
using FactCheck

macro title(s)
    return :("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ " * $s)
end
macro subtitle(s)
    return :("████████████████████████████████████████████████ " * $s)
end

include("testglm.jl")
end
