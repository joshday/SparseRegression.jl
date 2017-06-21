function name(a, withparams = false)
    s = replace(string(typeof(a)), "SparseRegression.", "")
    if !withparams
        s = replace(s, r"\{(.*)", "")
    end
    s
end

function header(io, s, ln::Bool = true)
    # print_with_color(:light_cyan, io, "■ $s")
    print(io, "■ $s")
    ln && println(io)
end
function print_item(io::IO, name::AbstractString, value, newline = true)
    print(io, "  >" * @sprintf("%15s", name * ":  "))
    print(io, value)
    newline && println(io)
end
