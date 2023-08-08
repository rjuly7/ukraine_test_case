#Importing libraries
import pandas as pd
import math
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import pandapower as pp 
import json 
import generator
import load
import line
import substation 
import bus 
from build_functions import *

#Defining Constants
EARTH_RADIUS = 6371 #km
SUBSTATION_BUS_DISTANCE = 4 #km 

epsilon = 8.85418782e-12*1000*1e9 #nanofarads/km 
mu_o = 4*np.pi*1e-7*1000  #H/km   
FREQUENCY = 50 #Hz
ANGLE_MAX = 30 * math.pi/180 #radians

def get_radius4(r,d):
    return np.power(r*np.sqrt(2)*np.power(d,3),1/4)

def get_radius3(r,d):
    return np.power(r*d*d,1/3)

def get_radius2(r,d):
    return np.sqrt(r*d)

#Table 4.1, Appendix A.4 of book
#ro, GMD, GMR in meters, r in ohms/km 
line_parameters = { 110: {"ro": 0.953/2*25.4/1000, "r": 0.168 * 0.621371, "GMD": 5, "GMR": 0.0328 * 0.3048, "b": 1},
                    220: {"ro": 1.108/2*25.4/1000, "r": 0.117 * 0.621371, "GMD": 7, "GMR": 0.0375 * 0.3048, "b": 1},
                    330: {"ro": get_radius2(1.545/2*25.4/1000, 45.7/100), "r": 0.059 * 0.621371, "GMD": 9, "GMR": get_radius2(0.052 * 0.3048, 45.7/100), "b": 2},
                    500: {"ro": get_radius3(1.293/2*25.4/1000, 45.7/100), "r": 0.0842 * 0.621371, "GMD": 11, "GMR": get_radius3(0.0435 * 0.3048, 45.7/100), "b": 3},
                    750: {"ro": get_radius4(1.293/2*25.4/1000, 45.7/100), "r": 0.0842 * 0.621371, "GMD": 14, "GMR": get_radius4(0.0435 * 0.3048, 45.7/100), "b": 4}}

np.random.seed(42)

#Reading CSV files
loads_csv = pd.read_csv('data_files/ukraine_cities_name_population_coordinates.csv')
generators_csv = pd.read_csv('data_files/ukraine_generators_power_capacities_coordinates.csv')
map_lines_csv = pd.read_csv('data_files/map_lines.csv')
pd.DataFrame.dropna(map_lines_csv,axis=1,how='all',inplace=True)
pd.DataFrame.dropna(map_lines_csv,axis=0,how='any',inplace=True)
map_substations_csv = pd.read_csv('data_files/substations.csv')
pd.DataFrame.dropna(map_substations_csv,axis=1,how='all',inplace=True)
pd.DataFrame.dropna(map_substations_csv,axis=0,how='any',inplace=True)

all_buses = [] 
bus_counter = 0
substation_counter = 0
all_substations = []
#First input substations from map and create buses corresponding to voltage levels
#These substations will not correspond to generator or load 
for i in range(map_substations_csv.shape[0]):
    #Get voltage levels at substation
    voltage_level = get_sub_voltage(map_substations_csv,map_lines_csv,i)
    #Create a new substation
    new_substation = substation.Substation(name = map_substations_csv.loc[i,'name'].strip(),
                                        latitude = map_substations_csv.loc[i,'latitude'],
                                        longitude = map_substations_csv.loc[i,'longitude'],
                                        voltage_level = voltage_level)
    new_substation.sub_list.append(new_substation.name)
    # #Add buses at each voltage level 
    all_substations.append(new_substation)
    substation_counter += 1

all_generators = []
for i in range(generators_csv.shape[0]):
    voltage_level = get_gen_voltage(generators_csv,map_lines_csv,i)
    #Add a generator entry corresponding to the maximum voltage level at the plant 
    new_generator = generator.Generator(name = generators_csv.loc[i,'generator_name'],
                                        min_real_power = 0, 
                                        max_real_power = generators_csv.loc[i, 'power_capacity_mv'],
                                        min_reactive_power = generators_csv.loc[i, 'power_capacity_mv'] * -0.5,
                                        max_reactive_power = generators_csv.loc[i, 'power_capacity_mv'] * 0.5,
                                        latitude = generators_csv.loc[i, 'latitude'],
                                        longitude = generators_csv.loc[i, 'longitude'],
                                        power_type = generators_csv.loc[i, 'power_type'],
                                        voltage_level = voltage_level)
    all_generators.append(new_generator)


all_loads = []
for i in range(loads_csv.shape[0]):
    voltage_level = get_load_voltage(loads_csv,map_lines_csv,i)
    #Add a load entry corresponding to the maximum voltage level 
    if not (np.isnan(loads_csv.loc[i,'population'])):
        new_load = load.Load(name = loads_csv.loc[i, 'city'], 
                            latitude = loads_csv.loc[i, 'lat'],
                            longitude = loads_csv.loc[i, 'long'],
                            population = loads_csv.loc[i, 'population'],
                            voltage_level = voltage_level)
        all_loads.append(new_load)

for i in range(len(all_generators)):
    g = all_generators[i]
    if g.substation == -1:
        new_substation = substation.Substation(name = g.name,
                                                latitude = g.latitude,
                                                longitude = g.longitude,
                                                voltage_level = g.voltage)
        new_substation.gen_list.append(g.name)
        #After creating new substation associated with generator, see if any other loads or generators are 
        #close enough to also be added to the new substation 
        for j in range(len(all_generators)):
            if all_generators[j].substation == -1:
                #Calculate distance between this generator and the newly created substation
                if get_distance(g,all_generators[j]) < SUBSTATION_BUS_DISTANCE:
                    all_generators[j].substation = substation_counter 
                    new_substation.voltage = new_substation.voltage.union(all_generators[j].voltage)
                    new_substation.gen_list.append(all_generators[j].name)
        if g.power_type == "Coal" or g.power_type == "Gas": #only coal/gas gens can be in same substation as a load 
            for j in range(len(all_loads)):
                if all_loads[j].substation == -1:
                    #Calculate distance between this load and the newly created substation
                    if get_distance(g,all_loads[j]) < SUBSTATION_BUS_DISTANCE:
                        all_loads[j].substation = substation_counter 
                        new_substation.voltage = new_substation.voltage.union(all_loads[j].voltage)
                        new_substation.load_list.append(all_loads[j].name)
        all_substations.append(new_substation)
        substation_counter += 1 

for i in range(len(all_loads)):
    l = all_loads[i]
    if l.substation == -1:
        new_substation = substation.Substation(name = l.name,
                            latitude = l.latitude,
                            longitude = l.longitude,
                            voltage_level = l.voltage)
        new_substation.load_list.append(l.name)
        #After creating new substation associated with load, see if any other loads are 
        #close enough to also be added to the new substation 
        for j in range(len(all_loads)):
            if all_loads[j].substation == -1:
                #Calculate distance between this bus and the newly created substation
                if get_distance(l,all_loads[j]) < SUBSTATION_BUS_DISTANCE:
                    all_loads[j].substation = substation_counter 
                    new_substation.voltage = new_substation.voltage.union(all_loads[j].voltage)
                    new_substation.load_list.append(all_loads[j].name)
        all_substations.append(new_substation)
        substation_counter += 1 

for i in range(len(all_substations)): 
    voltage_level = all_substations[i].voltage
    max_v = max(voltage_level)  #We assume generators at substation are connected to maximum voltage bus
    min_v = min(voltage_level)  #We assume loads at substation are connected to minimum voltage bus
    for v in voltage_level:
        new_bus = bus.Bus(name = all_substations[i].name + "_" + str(v), 
                        latitude =  all_substations[i].latitude,
                        longitude = all_substations[i].longitude,
                        voltage_level = v,
                        substation = i)
        if v == max_v:
            substation_gens = get_substation_loadgen(i,all_generators)
            for g_idx in substation_gens:
                all_generators[g_idx].gen_bus = bus_counter 
        if v == min_v:
            substation_loads = get_substation_loadgen(i,all_loads)
            for l_idx in substation_loads:
                all_loads[l_idx].load_bus = bus_counter 
        all_buses.append(new_bus)
        bus_counter += 1 

#Now add lines from the map 
all_lines = []
thermal_limits = []
map_lines = {750: [], 500: [], 330: [], 220: [], 110: []}
for i in range(map_lines_csv.shape[0]):
    f_sub_name = map_lines_csv.loc[i,'from'].strip()
    f_type = map_lines_csv.loc[i,'from_type'].strip()
    f_substation = find_substation_idx(all_substations,f_type,f_sub_name)
    t_sub_name = map_lines_csv.loc[i, 'to'].strip()
    t_type = map_lines_csv.loc[i,'to_type'].strip() 
    t_substation = find_substation_idx(all_substations,t_type,t_sub_name)
    line_voltage = map_lines_csv.loc[i,'voltage_kv']

    if f_substation == t_substation:
        continue 

    f_bus = find_bus_at_voltage(line_voltage,f_substation,all_buses)
    t_bus = find_bus_at_voltage(line_voltage,t_substation,all_buses)
    map_lines[line_voltage].append({f_bus,t_bus})

all_transformers, thermal_limits = add_transformers(all_substations,all_buses,thermal_limits)

net = create_line_net(all_buses,all_lines,all_loads,all_generators,all_transformers)
system_graph = find_system_connectivity(all_buses,all_lines,all_transformers)

all_lines, thermal_limits = add_lines_ranked(all_buses,all_lines,thermal_limits, 750,line_parameters,map_lines[750])

all_lines, thermal_limits = add_lines_ranked(all_buses,all_lines,thermal_limits, 500,line_parameters,map_lines[500])

all_lines, thermal_limits = add_lines_ranked(all_buses,all_lines,thermal_limits, 330,line_parameters,map_lines[330])

all_lines, thermal_limits = add_lines_ranked(all_buses,all_lines,thermal_limits, 220,line_parameters,map_lines[220])

all_lines, thermal_limits = add_lines_ranked(all_buses,all_lines,thermal_limits, 110,line_parameters,map_lines[110])

for i in range(len(all_buses)):
    count = 0
    #find lines to or from bus
    for j in range(len(all_lines)):
        l = all_lines[j]
        if l.from_bus == i or l.to_bus == i:
            count += 1
    if count == 0:
        print("ERROR: bus ", i, " NOT CONNECTED")

with open("data/full_limits.json", "w") as outfile:
    json.dump(thermal_limits, outfile)

import pickle 
all_data = {'bus': all_buses, 'gen': all_generators, 'load': all_loads, 'line': all_lines, 'substations': all_substations}

outfile=open("data/all_data.obj", "wb")
pickle.dump(all_data, outfile)
outfile.close()
#create empty net
net = pp.create_empty_network() 
net.f_hz = FREQUENCY 
net.sn_mva = 100 #base 100 MVA
#create buses
for b in all_buses:
    pp.create_bus(net, vn_kv=b.v_nom, max_vm_pu=b.v_max/b.v_nom, min_vm_pu=b.v_min/b.v_nom)
    
for l in all_loads:
    pp.create_load(net, bus=l.load_bus, p_mw = l.real_power_demanded, q_mvar = l.reactive_power_demanded)
    
for i in range(len(all_generators)):
    g = all_generators[i] 
    pp.create_gen(net, bus=g.gen_bus, p_mw = 0, vm_pu = 1.0, max_p_mw = g.max_real_power, 
                    min_p_mw=g.min_real_power, max_q_mvar=g.max_reactive_power, min_q_mvar=g.min_reactive_power)
    pp.create_poly_cost(net,i,"gen",cp1_eur_per_mw=g.costs)

for l in all_lines:
    pp.create_line_from_parameters(net, from_bus=l.from_bus, to_bus=l.to_bus, length_km=l.length, 
                  r_ohm_per_km=l.resistance, x_ohm_per_km=l.reactance, c_nf_per_km=l.capacitance, 
                  max_i_ka = l.max_i_ka) 

for t in all_transformers:
    pp.create_transformer_from_parameters(net, hv_bus = t.hv_bus, lv_bus = t.lv_bus, sn_mva=t.sn_mva, vn_hv_kv=t.hv, vn_lv_kv=t.lv, vk_percent=t.vk_percent, vkr_percent=t.vkr_percent, pfe_kw=t.pfe_kw, i0_percent=t.i0_percent)

net.gen["slack"][1] = True   #Burshtyn power station 

print("Number of generators: ", len(net.gen))

filename="test_case/ukraine_full.mat"
mpc = pp.converter.to_mpc(net,filename,init='flat')

#pp.rundcopp(net,verbose=True)

############################plot#######################
import pandas as pd
import matplotlib.pyplot as plt

x_values = []
y_values = []
voltages = []
for l in all_lines:
    f_bus = l.from_bus
    t_bus = l.to_bus 
    x_values.append([all_buses[f_bus].longitude,all_buses[t_bus].longitude])
    y_values.append([all_buses[f_bus].latitude,all_buses[t_bus].latitude])
    voltages.append(all_buses[f_bus].voltage)

gens_for_map = dict()
for i in range(len(all_generators)):
    g = all_generators[i]
    gens_for_map[i] = {'latitude' : all_buses[g.gen_bus].latitude, 'longitude' : all_buses[g.gen_bus].longitude}

loads_for_map = dict()
for i in range(len(all_loads)):
    l = all_loads[i]
    loads_for_map[i] = {'latitude' : all_buses[l.load_bus].latitude, 'longitude' : all_buses[l.load_bus].longitude}
gen_df = pd.DataFrame.from_dict(gens_for_map,orient='index')
load_df = pd.DataFrame.from_dict(loads_for_map,orient='index')

import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
street_map = gpd.read_file('data_files/stanford-nv937bq8361-shapefile/nv937bq8361.shp')
crs = {'init':"EPSG:4326"}
gen_geometry = [Point(xy) for xy in zip(gen_df['longitude'],gen_df['latitude'])]
gen_geo_df = gpd.GeoDataFrame(gen_df, crs=crs, geometry=gen_geometry)
load_geometry = [Point(xy) for xy in zip(load_df['longitude'],load_df['latitude'])]
load_geo_df = gpd.GeoDataFrame(load_df, crs=crs, geometry=load_geometry)

# create figure and axes, assign to subplot
fig, ax = plt.subplots(figsize=(15,15))

# add .shp mapfile to axes
street_map.plot(ax=ax, alpha=0.4,color='grey')
# add geodataframe to axes
# assign ‘price’ variable to represent coordinates on graph
# add legend
# make datapoints transparent using alpha
# assign size of points using markersize
gen_geo_df.plot(ax=ax,marker="D",markersize=5,label="Generators")
load_geo_df.plot(ax=ax,markersize=2,label="Loads")
vcolors = {750: "blue", 500: "red", 330: "yellow", 220: "green", 110: "black"}
vlabels = {750: "750", 500: "500", 330: "330", 220: "220", 110: "110"}
opacities = {750: 0.5, 500: 0.5, 330: 0.5, 220: 0.8, 110: 1}
linewidths = {750: 1, 500: 0.9, 330: 0.8, 220: 0.7, 110: 0.3}

for i in range(len(x_values)):
    plt.plot(x_values[i], y_values[i],color=vcolors[voltages[i]],linewidth=linewidths[voltages[i]],alpha=opacities[voltages[i]])
# add title to graph
handles, labels = ax.get_legend_handles_labels()

from matplotlib.lines import Line2D
plot_lines = []
plot_labels = []
for i in vlabels.keys():
    v = vlabels[i]
    plot_lines.append(Line2D([0], [0], label=v, color=vcolors[i]))
    plot_labels.append(v)

# add manual symbols to auto legend
handles.extend(plot_lines)
labels.extend(plot_labels)
ax.legend(handles, labels)

# add title to graph
plt.title('Transmission Network', fontsize=15,fontweight='bold')
# set latitiude and longitude boundaries for map display
# show map
plt.savefig("figs/network_plot.pdf")

plt.show()