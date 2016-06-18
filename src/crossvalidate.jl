abstract CrossValidation

immutable MCCV  <: CrossValidation
    m::Int                  # number of random datasets to fit
    train_prop::Float64     # proportion of data to be training data
end

default_measure(m::LinearRegression)        = :mse
default_measure(m::L1Regression)        = :mse
default_measure(m::LogisticRegression)  = :misclass
default_measure(m::PoissonRegression)   = :mse
default_measure(m::SVMLike)             = :misclass
default_measure(m::QuantileRegression)  = :loss
default_measure(m::HuberRegression)     = :mse


function crossvalidate!(
        o::SparseReg,
        cv::MCCV = MCCV(10, .7),
        measure::Symbol = default_measure(o.model);
        kw...
    )
    n, p = size(o.x)
    ntrain = round(Int, n * .7)

    for k in 1:cv.m
        rows_train = StatsBase.sample(1:n, ntrain)
        rows_test = setdiff(1:n, rows_train)
        # partition data into train and test
        xtrain  = o.x[rows_train, :]
        ytrain  = o.y[rows_train]
        xtest   = o.x[rows_test, :]
        ytest   = o.y[rows_test]
        # fit training data

        if measure == :mse

        elseif measure == :misclass

        elseif measure == :loss

        else
            error("measure = $measure is not an option")
        end
    end
end
