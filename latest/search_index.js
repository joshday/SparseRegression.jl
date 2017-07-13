var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "SparseRegression.jl",
    "title": "SparseRegression.jl",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#SparseRegression.jl-1",
    "page": "SparseRegression.jl",
    "title": "SparseRegression.jl",
    "category": "section",
    "text": "SparseRegression relies on primitives defined in the JuliaML ecosystem to implement high-performance algorithms for linear models which often produce sparsity in the coefficients.  "
},

{
    "location": "index.html#Objective-1",
    "page": "SparseRegression.jl",
    "title": "Objective",
    "category": "section",
    "text": "The objective functions that SparseRegression can solve are of the form:frac1nsum_i=1^n f(y_i x_i^Tbeta) + sum_j=1^p lambda_j J(beta_j)where f is a loss function and J is a penalty or regularization function.The three core JuliaML packages that SparseRegression brings together are:LossFunctions\n\"grammar of losses\"\nPenaltyFunctions\n\"grammar of regularization\"\nLearningStrategies\n\"grammar of iterative learning\"With few exceptions, SparseRegression can handle:any Loss from LossFunctions.jl\nany ElementPenalty from PenaltyFunctions.jl"
},

{
    "location": "smodel.html#",
    "page": "SModel",
    "title": "SModel",
    "category": "page",
    "text": ""
},

{
    "location": "smodel.html#SModel-1",
    "page": "SModel",
    "title": "SModel",
    "category": "section",
    "text": "The main struct exported by SparseRegression is SModel:struct SModel{L <: Loss, P <: Penalty}\n    β::Vector{Float64}\n    λfactor::Vector{Float64}\n    loss::L\n    penalty::P\nendAn SModel is constructed with the number of predictors (or Obs), as well as a loss, penalty, and λfactor in any order (and it's type stable).SModel(5)  # default: LinearRegression, L2Penalty(), fill(.1, 5)\nSModel(5, LogisticRegression(), L1Penalty())\nSModel(5, L2Penalty(), L1HingeLoss())\nSModel(obs, NoPenalty(), QuantileRegression(.7))After creating an SModel, it must then be learned with an Algorithm and any other number of learning strategies."
},

{
    "location": "smodel.html#Example-1",
    "page": "SModel",
    "title": "Example",
    "category": "section",
    "text": "using SparseRegression\n\nx, y = randn(1000, 10), randn(1000)\n\nobs = Obs(x, y)\n\ns = SModel(obs)\n\n# Learn the model using Proximal Gradient Method\n# - maximum of 50 iterations\n# - convergence criteria: norm(β - βold) < 1e-6\nlearn!(s, ProxGrad(obs), MaxIter(50), Converged(coef))"
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
    "text": "An Algorithm contains Obs, parameters for the algorithm, and storage buffers.  Some algorithms only work with specific loss/penalty combinations."
},

{
    "location": "algorithms.html#ProxGrad(obs,-s)-1",
    "page": "Algorithms",
    "title": "ProxGrad(obs, s)",
    "category": "section",
    "text": "Proximal Gradient Method with step size s.  Handles any loss and convex penalty."
},

{
    "location": "algorithms.html#Fista(obs,-s)-1",
    "page": "Algorithms",
    "title": "Fista(obs, s)",
    "category": "section",
    "text": "Fast Iterative Shrinkage-Thresholding Algorithm (accelerated proximal gradient) with step size s.  Handles any loss and convex penalty."
},

{
    "location": "algorithms.html#GradientDescent(obs,-s)-1",
    "page": "Algorithms",
    "title": "GradientDescent(obs, s)",
    "category": "section",
    "text": "Gradient Descent with step size s.  Handles any loss and penalty."
},

{
    "location": "algorithms.html#Sweep(obs)-1",
    "page": "Algorithms",
    "title": "Sweep(obs)",
    "category": "section",
    "text": "Linear or Ridge regression via the sweep operator."
},

{
    "location": "algorithms.html#LinRegCholesky(obs)-1",
    "page": "Algorithms",
    "title": "LinRegCholesky(obs)",
    "category": "section",
    "text": "Linear or Ridge regression via Cholesky decomposition"
},

{
    "location": "observations.html#",
    "page": "Observations",
    "title": "Observations",
    "category": "page",
    "text": ""
},

{
    "location": "observations.html#Observations-1",
    "page": "Observations",
    "title": "Observations",
    "category": "section",
    "text": "Observations are wrapped in a lightweight Obs typejulia> x, y = randn(1000, 10), randn(1000);\n\njulia> Obs(x,y)\nSparseRegression.Obs{Void,Float64,Array{Float64,2},Array{Float64,1}}\n  > x: 1000×10 Array{Float64,2}\n  > y: 1000-element Array{Float64,1}\n  > w: VoidOptionally, the observations can be given a weight vectorjulia> Obs(x, y, rand(1000))\nSparseRegression.Obs{Array{Float64,1},Float64,Array{Float64,2},Array{Float64,1}}\n  > x: 1000×10 Array{Float64,2}\n  > y: 1000-element Array{Float64,1}\n  > w: 1000-element Array{Float64,1}This allows algorithms to dispatch on whether or not observations are weighted."
},

]}
