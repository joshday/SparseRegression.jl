function name(a, withparams = false)
    s = replace(string(typeof(a)), "SparseRegression.", "")
    if !withparams
        s = replace(s, r"\{(.*)", "")
    end
    s
end

function header(io, s, ln::Bool = true)
    print_with_color(:light_cyan, io, "■ $s")
    ln && println(io)
end
function print_item(io::IO, name::AbstractString, value, newline = true)
    print(io, "  >" * @sprintf("%15s", name * ":  "))
    print(io, value)
    newline && println(io)
end
function print_items(io::IO, o, nms = fieldnames(o))
    for nm in nms
        print_item(io, "$nm", getfield(o, nm), nm != nms[end])
    end
end

#----------# Display fields like: (a = 1, b = 5.0, ...)
function printfields(io::IO, o, nms = fieldnames(o))
    if length(nms) != 0
        s = "("
        for nm in nms
            s *= "$nm = $(getfield(o, nm))"
            if nms[end] != nm
                s *= ", "
            end
        end
        s *= ")"
        return print(io, s)
    else
        return print(io, "")
    end
end

showme(o) = []
