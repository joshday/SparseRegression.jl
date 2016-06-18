using RecipesBase

@recipe function f(o::SparseReg)
    :label --> ["B_$j" for j in 1:size(o.β, 1)]'
    :xguide --> "lambda"
    :yguide --> "value"
    :title --> "Solution Path"
    o.λ, o.β'
end


@recipe function f{M <: BivariateModel}(o::SparseReg{M}, x::Matrix, y::Vector)
    J = size(o.β, 2)
    misclass, precision, recall, f1 = zeros(J), zeros(J), zeros(J), zeros(J)

    for j in 1:J
        yhat = classify(o, x, o.λ[j])
        misclass[j] = mean(y .== yhat)

        number_true_1 = sum(y .== yhat .== 1)
        precision[j] = number_true_1 / sum(yhat .== 1)
        recall[j] = number_true_1 / sum(y .== 1)
        f1[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    end
    :title --> "Model Diagnostics by lambda"
    :label --> ["Accuracy" "Precision" "Recall" "F1"]
    :yguide --> "value"
    :xguide --> "lambda"
    o.λ, hcat(misclass, precision, recall, f1)
end

@recipe function f(o::SparseReg, x::Matrix, y::Vector)
    J = size(o.β, 2)
    mse = zeros(J)
    for j in 1:J
        yhat = predict(o, x, o.λ[j])
        mse[j] = sqrt(mean(abs2(y - yhat)))
    end
    :title --> "Error by lambda"
    :yguide --> "MSE"
    :xguide --> "lambda"
    o.λ, mse
end
