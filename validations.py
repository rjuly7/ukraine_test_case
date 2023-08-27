# importing the module
import pickle
import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

infile = open('data/all_data.obj', 'rb')
all_data = pickle.load(infile)
infile.close()

all_lines = all_data['line']
all_buses = all_data['bus']
all_generators = all_data['gen']
all_loads = all_data['load']
all_substations = all_data['substations']

#Validations
validation_names = []
validation_descriptions = []
validation_statistics = []
validation_requirements = []
validation_realistic = []

#1. Number of Buses per Substation (Graph)
name = "Number_of_Buses_per_Substation"
graph_name = "Number of Buses per Substation"

num_buses_in_substation = np.array([])
for i in range(len(all_substations)):
    num_buses_in_substation = np.append(num_buses_in_substation, len(all_substations[i].voltage))

print("Average number of buses per substation: ", np.mean(num_buses_in_substation))

unique_num_buses_in_substation, bus_counts_in_substation = np.unique(num_buses_in_substation, return_counts=True)
frequencies = bus_counts_in_substation / float(len(all_substations))

plt.figure(figsize=(8,5))
plt.title(graph_name)
plt.xlabel("Number of Buses in a Substation")
plt.ylabel("Fraction of Substations")
plt.plot(unique_num_buses_in_substation, frequencies)

if(os.path.isfile(f'{name}.png')):
    os.remove(f'{name}.png')
plt.savefig('figs/'+name+'.png')
           
#2. Substation voltage levels (CSV)
name = "Substation Voltage Levels"
description = "Percent of substations containing voltage levels"
requirement = "85 - 100% low-voltage substations and 7 - 25% high-voltage substations"

low_voltage_substations = 0
high_voltage_substations = 0 

for i in range(len(all_substations)):
    if(110 in all_substations[i].voltage):
        low_voltage_substations += 1
    
    if(220 in all_substations[i].voltage or 330 in all_substations[i].voltage or 750 in all_substations[i].voltage):
        high_voltage_substations += 1

statistic = [float(low_voltage_substations) / len(all_substations) * 100, float(high_voltage_substations) / len(all_substations) * 100]
print("Substation voltages: ", statistic)
realistic = 7 <= float(statistic[0]) and float(statistic[0]) <= 25 and 85 < float(statistic[1])

validation_names.append(name)
validation_descriptions.append(description)
validation_statistics.append(statistic)
validation_requirements.append(requirement)
validation_realistic.append(realistic)

#3. Percent of substations containing load (CSV)
name = "Percent of substations containing load"
description = "Ratio of substations containing load to total substations"
requirement = "Between 75 - 90%"

substations_with_loads = np.zeros(len(all_substations))

for i in range(len(all_loads)):
    substations_with_loads[all_loads[i].substation] = 1

statistic = float(np.sum(substations_with_loads)) / len(all_substations) * 100
print("Percent of substations containing load: ", statistic)
realistic = 75 <= statistic and statistic <= 90

validation_names.append(name)
validation_descriptions.append(description)
validation_statistics.append(statistic)
validation_requirements.append(requirement)
validation_realistic.append(realistic)

#4. Load at Each Bus (Plot)
name = "Load_at_Each_Bus"
graph_name = "Load at Each Bus"

bus_loads = np.zeros(len(all_buses),dtype=float)
for i in range(len(all_buses)):
    for l in range(len(all_loads)):
        if(all_loads[l].load_bus == i):
            bus_loads[i] += all_loads[l].real_power_demanded
           
bus_loads = bus_loads[np.where(bus_loads > 0)]
print("Mean bus load: ", np.mean(bus_loads))

frequencies,bins = np.histogram(bus_loads,bins=20)
frequencies = frequencies/len(bus_loads)
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(8,5))
plt.plot(bins[0:-1],frequencies)
plt.title(graph_name)
plt.xlabel("Amount of Load at Bus (MW)")
plt.ylabel("Fraction of Buses")
plt.yscale("log")

if(os.path.isfile(f'{name}.png')):
    os.remove(f'{name}.png')
plt.savefig('figs/'+name+'.png')

#5. Ratio of Generator Capacity to Total Load (CSV)
name = "Ratio of Generator Capacity to Total Load"
description = "Ratio of the sum of maximum generator power production to the sum of load demanded"
requirement = "Between 1.20 - 1.60"

total_generator_max_power_production = 0
total_load_demanded = 0

for i in range(len(all_generators)):
    total_generator_max_power_production += all_generators[i].max_real_power

for i in range(len(all_loads)):
    total_load_demanded += all_loads[i].real_power_demanded

statistic = float(total_generator_max_power_production / total_load_demanded)
print("capacity to load ratio: ", statistic)
realistic = 1.20 <= statistic and statistic <= 1.60

validation_names.append(name)
validation_descriptions.append(description)
validation_statistics.append(statistic)
validation_requirements.append(requirement)
validation_realistic.append(realistic)

num_solar = sum(1 for g in all_generators if g.power_type == "Solar")
print("Solar: ", num_solar)
num_hydro = sum(1 for g in all_generators if g.power_type == "Hydro")
print("Hydro: ", num_hydro)
num_coal = sum(1 for g in all_generators if g.power_type == "Coal")
print("Coal: ", num_coal)
num_gas = sum(1 for g in all_generators if g.power_type == "Gas")
print("Gas: ", num_gas)
num_nuclear = sum(1 for g in all_generators if g.power_type == "Nuclear")
print("Nuclear: ", num_nuclear)

#6. Percent of substations containing generation (CSV)
name = "Percent of substations containing generation"
description = "Ratio of substations containing generators to total substations"
requirement = "Between 5 - 25%"

substations_with_generators = np.zeros(len(all_substations))

for i in range(len(all_generators)):
    substations_with_generators[all_generators[i].substation] = 1

statistic = float(np.sum(substations_with_generators)) / len(all_substations) * 100
print("Percent of substations containing generation: ", statistic)
realistic = 5 <= statistic and statistic <= 25

validation_names.append(name)
validation_descriptions.append(description)
validation_statistics.append(statistic)
validation_requirements.append(requirement)
validation_realistic.append(realistic)

#7. Capacities of Generators (Plot)
name = "Capacities_of_Generators"
graph_name = "Capacities of Generators"

bus_generators = np.zeros(len(all_buses),dtype=float)
for i in range(len(all_buses)):
    for l in range(len(all_generators)):
        if(all_generators[l].gen_bus == i):
            bus_generators[i] += all_generators[l].max_real_power
           
bus_generators = bus_generators[np.where(bus_generators > 0)]

frequencies,bins = np.histogram(bus_generators,bins=20)
frequencies = frequencies/len(bus_generators)

plt.figure(figsize=(8,5))
plt.plot(bins[0:-1],frequencies)
plt.title(graph_name)
plt.xlabel("Generator Size (MW)")
plt.ylabel("Probability (area)")
#plt.xscale("log")

count = 0
for i in range(len(all_generators)):
    g = all_generators[i]
    if g.max_real_power >= 25 and g.max_real_power <= 200:
        count += 1
print("Percent between 25,200: ", count/len(all_generators))

count = 0
for i in range(len(all_generators)):
    g = all_generators[i]
    if g.max_real_power >= 200:
        count += 1
print("Percent above 200: ", count/len(all_generators))

if(os.path.isfile(f'{name}.png')):
    os.remove(f'{name}.png')
plt.savefig('figs/'+name+'.png')

#14. Transmission Line X/R Ratio and MVA Limit (Table)
name = "Transmission_Line_X_by_R_Ratio_MVA_Limit_by_Voltage"

for i in range(len(all_lines)):
    if(all_lines[i].voltage == 110):
        X_by_R_110 = float(all_lines[i].reactance / all_lines[i].resistance)
        line_flow_limit_110 = all_lines[i].line_flow_limit
        break
for i in range(len(all_lines)):
    if(all_lines[i].voltage == 220):
        X_by_R_220 = float(all_lines[i].reactance / all_lines[i].resistance)
        line_flow_limit_220 = all_lines[i].line_flow_limit
        break 
for i in range(len(all_lines)):
    if(all_lines[i].voltage == 330):
        X_by_R_330 = float(all_lines[i].reactance / all_lines[i].resistance)
        line_flow_limit_330 = all_lines[i].line_flow_limit
        break
for i in range(len(all_lines)):
    if(all_lines[i].voltage == 750):
        X_by_R_750 = float(all_lines[i].reactance / all_lines[i].resistance)
        line_flow_limit_750 = all_lines[i].line_flow_limit
        break

X_by_R_line_flow_limit_table = {"Voltage Level (KV)": np.array([110, 220, 330, 750], dtype = int),
                "X/R": np.array([X_by_R_110, X_by_R_220, X_by_R_330, X_by_R_750], dtype = float),
                "Line Flow Limit": np.array([line_flow_limit_110, line_flow_limit_220, line_flow_limit_330, line_flow_limit_750], dtype = float)}
X_by_R_line_flow_limit_display = pd.DataFrame(X_by_R_line_flow_limit_table)

if(os.path.isfile(f'{name}.csv')):
    os.remove(f'{name}.csv')
X_by_R_line_flow_limit_display.to_csv('data/'+name+'.csv')

print()
print("Line reactances, p.u.: ")
lvs = [110,220,330,500,750]
criteria = {110: [0.006387, 0.000796], 220: [ 0.001550, 0.000343], 330: [0.000518, 0.000198], 500: [0.000210, 0.000121], 750: [-10,10]}
for vv in lvs:
    zbase = vv**2/100 
    for i in range(len(all_lines)):
        l = all_lines[i]
        if l.voltage == vv:
            print(vv, l.reactance/zbase)
            if l.reactance/zbase > criteria[vv][1] and l.reactance/zbase < criteria[vv][0]:
                print(vv, "within range!")
            break  

print()
print("X/R ratios:")
criteria = {110: [8.3,2.5], 220: [12.5,6.4], 330: [16,9], 500: [26,11], 750: [-100,100]}
for vv in lvs:
    for i in range(len(all_lines)):
        l = all_lines[i]
        if l.voltage == vv:
            print(vv, l.reactance/l.resistance)
            if l.reactance/l.resistance > criteria[vv][1] and l.reactance/l.resistance < criteria[vv][0]:
                print(vv, "within range!")
            break 

print()
print("MVA ratings:")
criteria = {110: [255,92], 220: [797,327], 330: [1494,897], 500: [3464, 1732], 750: [0, 10000]}
for vv in lvs:
    rc = 0
    lc = 0
    for i in range(len(all_lines)):
        l = all_lines[i]
        if l.voltage == vv:
            lc += 1
            lrate = l.line_flow_limit
            if lrate > criteria[vv][1] and lrate < criteria[vv][0]:
                rc += 1
    print(vv, rc/lc)