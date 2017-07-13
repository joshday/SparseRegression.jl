# Observations
- Observations are wrapped in a lightweight `Obs` type
```julia
julia> x, y = randn(1000, 10), randn(1000);

julia> Obs(x,y)
SparseRegression.Obs{Void,Float64,Array{Float64,2},Array{Float64,1}}
  > x: 1000×10 Array{Float64,2}
  > y: 1000-element Array{Float64,1}
  > w: Void
```
- Optionally, the observations can be given a weight vector
```julia
julia> Obs(x, y, rand(1000))
SparseRegression.Obs{Array{Float64,1},Float64,Array{Float64,2},Array{Float64,1}}
  > x: 1000×10 Array{Float64,2}
  > y: 1000-element Array{Float64,1}
  > w: 1000-element Array{Float64,1}
```
- This allows algorithms to dispatch on whether or not observations are weighted.
