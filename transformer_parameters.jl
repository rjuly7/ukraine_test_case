using Pkg
Pkg.activate(".")
using PowerModels 
using Ipopt 
using Statistics 
using Random, Distributions

Random.seed!(123) # Setting the seed

PowerModels.silence() 

case_path = "test_case/ukraine_full.m"
data = parse_file(case_path);

function find_transformers(lv,hv)
    trafos = []
    for (i,br) in data["branch"]
        f_bus = string(br["f_bus"])
        t_bus = string(br["t_bus"])
        fv = data["bus"][f_bus]["base_kv"]
        tv = data["bus"][t_bus]["base_kv"]
        if (fv == lv && tv == hv) || (fv == hv && tv == lv)
            push!(trafos, i)
        end
    end
    return trafos
end

x_dist = Normal(0.125,0.04)

voltages = Dict(110 => [220, 330], 220 => [330,500], 330 => [500,750], 500 => [750])
rate_dists = Dict(220 => Normal(203,90), 330 => Normal(444,180), 500 => Normal(812,300), 750 => Normal(1200,400))
xr_ratio_dists = Dict(220 => Normal(44,13), 330 => Normal(60,17), 500 => Normal(70,18), 750 => Normal(85,19))
rate_means = Dict(220 => [63,203,470], 330 => [200,444,702], 500 => [215,812,1383], 750 => [240,1200,1500])
xr_ratio_means = Dict(220 => [25,44,84], 330 => [35,60,157], 500 => [44,70,119], 750 => [54,85,130])

for lv in keys(voltages)
    for hv in voltages[lv]
        trafos = find_transformers(lv,hv)
        for i in trafos
            rate = -1
            while rate < 50
                rate = rand(rate_dists[hv])
            end
            data["branch"][i]["rate_a"] = rate/data["baseMVA"]
            x = -1 
            while x < 0
                x = rand(x_dist)
            end
            data["branch"][i]["br_x"] = x/data["branch"][i]["rate_a"]
            xr_ratio = -1
            while xr_ratio < 0
                xr_ratio = rand(xr_ratio_dists[hv])
            end
            data["branch"][i]["br_r"] = data["branch"][i]["br_x"]/xr_ratio
        end
    end
end

for lv in keys(voltages)
    for hv in voltages[lv]
        trafos = find_transformers(lv,hv)
        for i in trafos
            rate = data["branch"][i]["rate_a"]*data["baseMVA"]
            x = data["branch"][i]["br_x"]*data["branch"][i]["rate_a"]
            xr_ratio = data["branch"][i]["br_x"]/data["branch"][i]["br_r"] 
            if rate < rate_means[hv][1] || rate > rate_means[hv][3]
                println("Rate ", i, ", ", rate)
                println(rate_means[hv])
                println() 
            end
            if x < 0.05 || x > 0.2
                println("X ", i, ", ", x)
                println()
            end
            if xr_ratio < xr_ratio_means[hv][1] || xr_ratio > xr_ratio_means[hv][3]
                println("XR_ratio ", i, ", ", xr_ratio)
                println(xr_ratio_means[hv])
                println()
            end
        end
    end
end

#Save the case with transformer reactance values from distribution 
export_matpower(case_path,data)
#Reload case with new reactance values 
data = parse_file(case_path);

#Now remove mva limits on trafos to solve ac opf
for lv in keys(voltages)
    for hv in voltages[lv]
        trafos = find_transformers(lv,hv)
        for i in trafos
            data["branch"][i]["rate_a"] = 10000
        end
    end
end

# for i in keys(data["branch"])
#     data["branch"][i]["rate_a"] = 10000
# end

# for i in keys(data["gen"])
#     data["gen"][i]["qmin"] = -10000
#     data["gen"][i]["qmax"] = 10000
# end

for i in keys(data["load"])
    data["load"][i]["pd"] = data["load"][i]["pd"]
    data["load"][i]["qd"] = data["load"][i]["qd"]
end

pm = instantiate_model(data, ACPPowerModel, PowerModels.build_opf)
#print(pm.model)

result = optimize_model!(pm, optimizer=Ipopt.Optimizer)


data = parse_file(case_path);
diff = Dict()
for (i,br) in result["solution"]["branch"]
    s_br = max(sqrt(br["pf"]^2 + br["qf"]^2), sqrt(br["pt"]^2 + br["qt"]^2))
    diff[i] = s_br - data["branch"][i]["rate_a"]
end

for (i,d) in diff
    if d > 0
        println(i, ", ", d, "  lim: ", data["branch"][i]["rate_a"])
        br = data["branch"][i]
        old_rate = br["rate_a"]
        data["branch"][i]["rate_a"] = (br["rate_a"] + d)*1.1
        #data["branch"][i]["br_x"] = br["br_x"]*old_rate/data["branch"][i]["rate_a"]
        f_bus = string(br["f_bus"])
        t_bus = string(br["t_bus"])
        fv = data["bus"][f_bus]["base_kv"]
        tv = data["bus"][t_bus]["base_kv"]
        hv = 0 
        if fv > tv
            hv = fv
        else
            hv = tv 
        end
        xr_ratio = -1
        while xr_ratio < 0
            xr_ratio = rand(xr_ratio_dists[hv])
        end
        #data["branch"][i]["br_r"] = br["br_x"]/xr_ratio  
        println(data["branch"][i]["rate_a"],data["branch"][i]["br_x"],data["branch"][i]["br_r"])      
    end
end

export_matpower(case_path,data)

data = parse_file(case_path);

pm = instantiate_model(data, ACPPowerModel, PowerModels.build_opf)
#print(pm.model)

result = optimize_model!(pm, optimizer=Ipopt.Optimizer)

xs = [] 

for lv in keys(voltages)
    for hv in voltages[lv]
        trafos = find_transformers(lv,hv)
        for i in trafos
            push!(xs, data["branch"][i]["br_x"]*data["branch"][i]["rate_a"])
        end
    end
end

function print_stuff(lval,mval,hval,my_list)
    println("Median: ", median(my_list))
    println("10%: ", quantile!(my_list,0.1))
    println("90%: ", quantile!(my_list,0.9))
    N = size(my_list)[1]
    in_range = findall(x->(x>lval && x<hval), my_list)
    println("Percent in range: ", size(in_range)[1]/N)
    println("Percent below median: ", size(findall(x->x<mval,my_list))[1]/N)
    println("Percent above median: ", size(findall(x->x>mval,my_list))[1]/N)
    println() 
end

print_stuff(0.05,0.12,0.2,xs)

other_stats = Dict(i => Dict("xs" => [], "xr_ratios" => [], "rates" => []) for i in keys(rate_dists))

for lv in keys(voltages)
    for hv in voltages[lv]
        trafos = find_transformers(lv,hv)
        xr_ratios = [] 
        rates = []
        xs = []
        for i in trafos
            br = data["branch"][i]
            push!(xr_ratios, br["br_x"]/br["br_r"])
            push!(rates, br["rate_a"])
            push!(xs, br["br_x"]*br["rate_a"])
        end
        append!(other_stats[hv]["xr_ratios"],xr_ratios)
        append!(other_stats[hv]["rates"],rates)
        append!(other_stats[hv]["xs"], xs)
    end
end

for hv in keys(other_stats)
    println(hv, "   N=", length(other_stats[hv]["rates"]))
    println("XR ratio: ")
    print_stuff(xr_ratio_means[hv][1],xr_ratio_means[hv][2],xr_ratio_means[hv][3],other_stats[hv]["xr_ratios"])
    println("Rates: ")
    print_stuff(rate_means[hv][1]/data["baseMVA"],rate_means[hv][2]/data["baseMVA"],rate_means[hv][3]/data["baseMVA"],other_stats[hv]["rates"])
    println("Xs: ")
    print_stuff(0.05,0.12,0.2,other_stats[hv]["xs"])
end


























