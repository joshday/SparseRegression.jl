var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Introduction-1",
    "page": "Introduction",
    "title": "Introduction",
    "category": "section",
    "text": "SparseRegression is a Julia package which combines JuliaML primitives to implement high-performance algorithms for fitting linear models."
},

{
    "location": "index.html#Objective-Function-1",
    "page": "Introduction",
    "title": "Objective Function",
    "category": "section",
    "text": "The objective function that SparseRegression can solve takes the form:frac1nsum_i=1^n w_i f(y_i x_i^Tbeta) + sum_j=1^p lambda_j J(beta_j)where f is a loss function, J is a penalty or regularization function, the w_i's are nonnegative observation weights and the lambda_j's are nonnegative element-wise regularization parameters.  Many models take this form:Model f(y_i x_i^Tbeta) g(beta_j)\nLasso Regression frac12(y_i - x_i^Tbeta)^2 beta_j\nRidge Regression frac12(y_i - x_i^Tbeta)^2 beta_j^2\nSVM max(0 1 - y_i x_i^Tbeta) beta_j^2"
},

{
    "location": "index.html#[JuliaML](https://github.com/JuliaML)-1",
    "page": "Introduction",
    "title": "JuliaML",
    "category": "section",
    "text": "The three core JuliaML packages that SparseRegression brings together are:LossFunctions\nPenaltyFunctions\nLearningStrategies"
},

{
    "location": "usage.html#",
    "page": "Usage",
    "title": "Usage",
    "category": "page",
    "text": ""
},

{
    "location": "usage.html#Usage-1",
    "page": "Usage",
    "title": "Usage",
    "category": "section",
    "text": "Create a model\nFit the model"
},

{
    "location": "usage.html#SparseRegression.SModel",
    "page": "Usage",
    "title": "SparseRegression.SModel",
    "category": "Type",
    "text": "SModel(x, y, args...)\n\nCreate a SparseRegression model with predictor AbstractMatrix x and response AbstractVector y.  x must have methods: \n\nA_mul_B!(::Matrix{Float64}, x, ::Vector{Float64})\nAt_mul_B!(::Matrix{Float64}, x, ::Vector{Float64})\n\nThe additional arguments can be given in any order.\n\nArguments\n\nloss::Loss = .5 * L2DistLoss()\npenalty::Penalty = L2Penalty()\nλ::Vector{Float64} = fill(size(x, 2), .1)\nw::Union{Void, AbstractWeights} = nothing\n\nExample\n\nx = randn(1000, 5)\ny = x * linspace(-1, 1, 5) + randn(1000)\ns = SModel(x, y)\nlearn!(s)\ns\n\n\n\n"
},

{
    "location": "usage.html#SModel-1",
    "page": "Usage",
    "title": "SModel",
    "category": "section",
    "text": "The model type used by SparseRegression is SModel.  An SModel holds onto the sufficient information for generating a solution fo the SparseRegression objective.note: Note\nConstructing an SModel does not create a fitted model.  It must be learn!-ed.SModel"
},

{
    "location": "usage.html#[LearningStrategies](https://github.com/JuliaML/LearningStrategies.jl)-1",
    "page": "Usage",
    "title": "LearningStrategies",
    "category": "section",
    "text": "An SModel can be learned with the default learning strategy with learn!(model).  You can provide more control over the learning process by providing your own LearningStrategy.SparseRegression implements several Algorithm <: LearningStrategy types to do SModel fitting.  An Algorithm must be constructed with an SModel to ensure storage buffers are the correct size.using SparseRegression\n\n# Make some fake data\nx = randn(1000, 10)\ny = x * linspace(-1, 1, 10) + randn(1000)\n\n# Create an SModel\ns = SModel(x, y)\n\n# All of the following are valid ways to calculate a solution\nlearn!(s)\nlearn!(s, strategy(ProxGrad(s), MaxIter(25), TimeLimit(.5)))\nlearn!(s, Sweep(s))\nlearn!(s, LinRegCholesky(s))"
},

{
    "location": "algorithms.html#",
    "page": "Algorithms",
    "title": "Algorithms",
    "category": "page",
    "text": ""
},

{
    "location": "algorithms.html#Algorithms-1",
    "page": "Algorithms",
    "title": "Algorithms",
    "category": "section",
    "text": "The first argument of an Algorithm's constructor is an SModel.  This is to ensure storage buffers are the correct size."
},

{
    "location": "algorithms.html#SparseRegression.ProxGrad",
    "page": "Algorithms",
    "title": "SparseRegression.ProxGrad",
    "category": "Type",
    "text": "ProxGrad(model, step = 1.0)\n\nProximal gradient method with step size step.  Works for any loss and any penalty with a prox method.\n\nExample\n\nx, y, β = SparseRegression.fakedata(L2DistLoss(), 1000, 10)\ns = SModel(x, y, L2DistLoss())\nstrat = strategy(MaxIter(50), ProxGrad(s))\nlearn!(s, strat)\n\n\n\n"
},

{
    "location": "algorithms.html#ProxGrad-1",
    "page": "Algorithms",
    "title": "ProxGrad",
    "category": "section",
    "text": "ProxGrad"
},

{
    "location": "algorithms.html#SparseRegression.Fista",
    "page": "Algorithms",
    "title": "SparseRegression.Fista",
    "category": "Type",
    "text": "Fista(model, step = 1.0)\n\nAccelerated proximal gradient method.  Works for any loss and any penalty with a prox method.\n\n\n\n"
},

{
    "location": "algorithms.html#Fista-1",
    "page": "Algorithms",
    "title": "Fista",
    "category": "section",
    "text": "Fista"
},

{
    "location": "algorithms.html#SparseRegression.AdaptiveProxGrad",
    "page": "Algorithms",
    "title": "SparseRegression.AdaptiveProxGrad",
    "category": "Type",
    "text": "AdaptiveProxGrad(s, divisor = 1.5, init = 1.0)\n\nProximal gradient method with adaptive step sizes.  AdaptiveProxGrad uses element-wise  learning rates.  Every time the sign of a coefficient switches, the step size for that coefficient is divided by divisor.\n\n\n\n"
},

{
    "location": "algorithms.html#AdaptiveProxGrad-1",
    "page": "Algorithms",
    "title": "AdaptiveProxGrad",
    "category": "section",
    "text": "AdaptiveProxGrad"
},

{
    "location": "algorithms.html#SparseRegression.GradientDescent",
    "page": "Algorithms",
    "title": "SparseRegression.GradientDescent",
    "category": "Type",
    "text": "GradientDescent(model, step = 1.0)\n\nGradient Descent.  Works for any loss and any penalty.\n\nExample\n\nx, y, β = SparseRegression.fakedata(L2DistLoss(), 1000, 10)\ns = SModel(x, y, L2DistLoss())\nstrat = strategy(MaxIter(50), GradientDescent(s))\nlearn!(s, strat)\n\n\n\n"
},

{
    "location": "algorithms.html#GradientDescent-1",
    "page": "Algorithms",
    "title": "GradientDescent",
    "category": "section",
    "text": "GradientDescent"
},

{
    "location": "algorithms.html#SparseRegression.Sweep",
    "page": "Algorithms",
    "title": "SparseRegression.Sweep",
    "category": "Type",
    "text": "Sweep(model)\n\nLinear/ridge regression via sweep operator.  Works for (scaled) L2DistLoss with NoPenalty or L2Penalty.  The Sweep algorithm has a closed form solution and is complete after one iteration.  It therefore doesn't need additional learning strategies such as MaxIter, Converged, etc.\n\nExample\n\nx, y, β = SparseRegression.fakedata(L2DistLoss(), 1000, 10)\ns = SModel(x, y, L2DistLoss())\nlearn!(s, Sweep(s))\n\n\n\n"
},

{
    "location": "algorithms.html#Sweep-1",
    "page": "Algorithms",
    "title": "Sweep",
    "category": "section",
    "text": "Sweep"
},

{
    "location": "algorithms.html#SparseRegression.LinRegCholesky",
    "page": "Algorithms",
    "title": "SparseRegression.LinRegCholesky",
    "category": "Type",
    "text": "LinRegCholesky(model)\n\nLinear/ridge regression via cholesky decomposition.  Works for (scaled) L2DistLoss with NoPenalty or L2Penalty.  The LinRegCholesky algorithm has a closed form solution  and is complete after one iteration.  It therefore doesn't need additional learning strategies such as MaxIter, Converged, etc.\n\nExample\n\nx, y, β = SparseRegression.fakedata(L2DistLoss(), 1000, 10)\ns = SModel(x, y, L2DistLoss())\nlearn!(s, Sweep(s))\n\n\n\n"
},

{
    "location": "algorithms.html#LinRegCholesky-1",
    "page": "Algorithms",
    "title": "LinRegCholesky",
    "category": "section",
    "text": "LinRegCholesky"
},

{
    "location": "algorithms.html#SparseRegression.LineSearch",
    "page": "Algorithms",
    "title": "SparseRegression.LineSearch",
    "category": "Type",
    "text": "LineSearch(algorithm)\n\nUse a line search in the update! of algorithm.  Currently, ProxGrad, Fista, and GradientDescent are supported.\n\nExample\n\nx, y, β = SparseRegression.fakedata(L2DistLoss(), 1000, 10)\ns = SModel(x, y, L2DistLoss())\nstrat = strategy(MaxIter(50), LineSearch(ProxGrad(s)))\nlearn!(s, strat)\n\n\n\n"
},

{
    "location": "algorithms.html#LineSearch-1",
    "page": "Algorithms",
    "title": "LineSearch",
    "category": "section",
    "text": "LineSearch"
},

]}
