using Pkg
Pkg.activate(".")
using PowerModels 
using Ipopt 
using Statistics 

PowerModels.silence() 

case_path = "test_case/ukraine_full.m";
data = parse_file(case_path);

pm = instantiate_model(data, ACPPowerModel, PowerModels.build_opf)

result = optimize_model!(pm, optimizer=Ipopt.Optimizer)

n_committed = 0
dispatch_over_80 = 0
for i in keys(data["gen"])
    pg = result["solution"]["gen"][i]["pg"]
    if pg > 1e-4 
        global n_committed += 1
        if pg/data["gen"][i]["pmax"] > 0.8
            global dispatch_over_80 += 1
        end
    end
end

println("Percent committed: ", n_committed/length(data["gen"]))

println("Percent dispatch over 80: ", dispatch_over_80/length(data["gen"]))