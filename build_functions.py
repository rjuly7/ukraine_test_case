import math 
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix 
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
import line
import transformer 
import pandapower as pp 

LOAD_BUS = 1
GENERATOR_BUS = 2
SUBSTATION_BUS = 3
LINE_MAX = 100 
EARTH_RADIUS = 6371 #km
LINES_FACTOR = 1.37
#Outer radius ro = 1.1080 in
epsilon = 8.85418782e-12*1000*1e9 #nanofarads/km 
mu_o = 4*np.pi*1e-7*1000  #H/km   
FREQUENCY = 50 #Hz
costs = {"Hydro": [0.9, 0.1], "Nuclear": [7.2504, 0.7534], "Coal": [24.7919, 8.0866], "Solar": [0, 0], "Gas": [34.2731, 10.9810]}
ANGLE_MAX = 30 * math.pi/180 #radians

sub_line_ratios = {110: 1.37, 220: 1.37, 330: 1.3, 500: 1.22, 750: 1.22}

transformer_parameters = { 110: {220: {"sn_mva": 1600, "vk_percent": 12, "vkr_percent": 0.26, "pfe_kw": 55, "i0_percent": 0.06}, 330: {"sn_mva": 1600, "vk_percent": 12.1, "vkr_percent": 0.26, "pfe_kw": 57, "i0_percent": 0.06}}, 
                          220: {330: {"sn_mva": 1600, "vk_percent": 12.2, "vkr_percent": 0.25, "pfe_kw": 60, "i0_percent": 0.06}, 500: {"sn_mva": 1600, "vk_percent": 12.2, "vkr_percent": 0.25, "pfe_kw": 60, "i0_percent": 0.06}},
                          330: {500: {"sn_mva": 1600, "vk_percent": 12.2, "vkr_percent": 0.25, "pfe_kw": 60, "i0_percent": 0.06}, 750: {"sn_mva": 1600, "vk_percent": 14, "vkr_percent": 0.22, "pfe_kw": 75, "i0_percent": 0.06}},
                          500: {750: {"sn_mva": 1600, "vk_percent": 14, "vkr_percent": 0.22, "pfe_kw": 75, "i0_percent": 0.06}}}

def get_substation_bus_at_voltage(sub_idx, voltage, all_buses):
    for i in range(len(all_buses)):
        if all_buses[i].substation == sub_idx and all_buses[i].voltage == voltage: 
            return i 

def get_substation_loadgen(substation, loadgen_list):
    loadgens = []
    for i in range(len(loadgen_list)):
        if loadgen_list[i].substation == substation:
            loadgens.append(i)
    return loadgens  

def get_distance(a, b):
    ilat = math.radians(a.latitude)
    ilong = math.radians(a.longitude)
    jlat = math.radians(b.latitude)
    jlong = math.radians(b.longitude)

    dlong = ilong - jlong
    dlat = ilat - jlat

    dist = np.power(math.sin(dlat / 2), 2) + math.cos(ilat) * math.cos(jlat) * pow(math.sin(dlong / 2), 2)
    dist = 2 * math.asin(math.sqrt(dist))
    dist = dist * EARTH_RADIUS
    return dist 

def find_substation_idx(all_substations,station_type,sub_name):
    idx = 20000
    for i in range(len(all_substations)):
        if station_type == "Generator":
            if sub_name in all_substations[i].gen_list:
                idx = i
        elif station_type == "City":
            if sub_name in all_substations[i].load_list:
                idx = i
        elif station_type == "Substation":
            if sub_name in all_substations[i].sub_list:
                idx = i
        else:
            print("Error: undefined type: ", station_type)
    return idx 

def find_bus_at_voltage(voltage,sub_idx,all_buses):
    idx = 20000
    for i in range(len(all_buses)):
        if all_buses[i].voltage == voltage and all_buses[i].substation == sub_idx:
            idx = i
    return idx 

def gen_is_in_map(generators_csv, map_lines_csv, idx):
    in_map = []
    name = generators_csv.loc[idx, 'generator_name'].strip() 
    for i in range(map_lines_csv.shape[0]):
        if map_lines_csv.loc[i,'from'].strip() == name and map_lines_csv.loc[i,'from_type'].strip() == 'Generator':
            in_map.append(i)
        if map_lines_csv.loc[i,'to'].strip() == name and map_lines_csv.loc[i,'to_type'].strip() == 'Generator':
            in_map.append(i)
    
    return in_map 

def load_is_in_map(loads_csv, map_lines_csv, idx):
    in_map = []
    name = loads_csv.loc[idx, 'city'].strip() 
    for i in range(map_lines_csv.shape[0]):
        if map_lines_csv.loc[i,'from'].strip()  == name and map_lines_csv.loc[i,'from_type'].strip()  == 'City':
            in_map.append(i)
        if map_lines_csv.loc[i,'to'].strip()  == name and map_lines_csv.loc[i,'to_type'].strip()  == 'City':
            in_map.append(i)
    
    return in_map 

def sub_is_in_map(subs_csv, map_lines_csv, idx):
    in_map = []
    name = subs_csv.loc[idx, 'name'].strip()
    for i in range(map_lines_csv.shape[0]):
        if map_lines_csv.loc[i,'from'].strip() == name and map_lines_csv.loc[i,'from_type'].strip() == 'Substation':
            in_map.append(i)
        if map_lines_csv.loc[i,'to'].strip() == name and map_lines_csv.loc[i,'to_type'].strip() == 'Substation':
            in_map.append(i)
    
    return in_map 

def get_gen_voltage(generators_csv, map_lines_csv, idx):
    in_map = gen_is_in_map(generators_csv, map_lines_csv, idx)
    voltage_level = set() 
    if len(in_map) > 0:
        for entry in in_map:
            voltage_level.add(map_lines_csv.loc[entry,'voltage_kv'])
        if min(voltage_level) >= 330:
            voltage_level.add(np.random.choice([110,220]))
    else:
        power_type = generators_csv.loc[idx,'power_type']
        if power_type == "Nuclear":
            voltage_level.add(-1)
            print("Error: nuclear plant not found on map")
        elif (power_type == "Coal" or power_type == "Gas"):
            voltage_level.add(220)
        elif power_type == "Hydro":
            voltage_level.add(220)
            voltage_level.add(110)
        elif power_type == "Solar":
            #voltage_level.add(220)
            voltage_level.add(110)
    return voltage_level 

def get_load_voltage(loads_csv, map_lines_csv, idx):
    in_map = load_is_in_map(loads_csv, map_lines_csv, idx)
    voltage_level = set() 
    if len(in_map) > 0:
        for entry in in_map:
            voltage_level.add(map_lines_csv.loc[entry,'voltage_kv'])
        if min(voltage_level) >= 330:
            voltage_level.add(np.random.choice([110,220]))
    else:
        voltage_level.add(110)
    if loads_csv.loc[idx,'population'] >= 0.8*1e4:
        voltage_level.add(220)
    return voltage_level 

def get_sub_voltage(map_substations_csv, map_lines_csv, idx):
    name = map_substations_csv.loc[idx, 'name'].strip()
    print(name)
    in_map = sub_is_in_map(map_substations_csv, map_lines_csv, idx)
    voltage_level = set() 
    if len(in_map) > 0:
        for entry in in_map:
            voltage_level.add(map_lines_csv.loc[entry,'voltage_kv'])
        if min(voltage_level) == 330:
            #voltage_level.add(np.random.choice([110,220]))
            voltage_level.add(110)
            voltage_level.add(220)
        elif min(voltage_level) == 500:
            print("here")
            voltage_level.add(220)
            voltage_level.add(110)
    else:
        print("Error: substation not found in map_lines")
        print(idx)
        print() 
    return voltage_level 

def get_buses_at_voltage(all_buses,voltage):
    bus_list = []
    for i in range(len(all_buses)):
        if all_buses[i].voltage == voltage:
            bus_list.append(i)
    return bus_list 

def find_line(all_lines,f_bus,t_bus,voltage):
    idx = -1
    for i in range(len(all_lines)):
        if all_lines[i].from_bus == f_bus and all_lines[i].to_bus == t_bus and all_lines[i].voltage == voltage:
            idx = i
        elif all_lines[i].from_bus == t_bus and all_lines[i].to_bus == f_bus and all_lines[i].voltage == voltage:
            idx = i
    return idx 

def find_line_all(all_lines,f_bus,t_bus):
    idx = -1
    for i in range(len(all_lines)):
        if all_lines[i].from_bus == f_bus and all_lines[i].to_bus == t_bus:
            idx = i
        elif all_lines[i].from_bus == t_bus and all_lines[i].to_bus == f_bus:
            idx = i
    return idx 

def get_lines_at_voltage(all_lines,voltage):
    line_list = []
    for i in range(len(all_lines)):
        if all_lines[i].voltage == voltage:
            line_list.append(i)
    return line_list 

def calc_line_parameters(params,line_voltage,line_length):
    R = params["r"]/params["b"]
    X = 2*np.pi*FREQUENCY * (mu_o/(2*np.pi)) * np.log(params["GMD"]/params["GMR"]) #ohms/km
    #B = 2*np.pi*FREQUENCY*(2*math.pi*epsilon/(np.log(params["GMD"]/params["ro"])))*1e-9 #Hz*Farads/km, 1e9 converts from nanoFarads to Farads
    C = (2*np.pi*epsilon/(np.log(params["GMD"]/params["ro"]))) #this is in nanoFarads/km like pandapower wants
    B = 2*np.pi*FREQUENCY*C*1e-9  
    z = complex(R*line_length,X*line_length)
    y = 1/z + complex(0,B*line_length/2)
    y_magnitude = np.abs(y)

    v_max_i = line_voltage
    v_max_j = line_voltage 
    #Thermal limit (MVA) on power from Carleton's paper:
    line_flow_limit = np.sqrt(np.square(v_max_i) * np.square(y_magnitude) * (np.square(v_max_i) + np.square(v_max_j) - 2 * v_max_i * v_max_j * np.cos(ANGLE_MAX)))
    #pandapower assumes vi=vj:
    #max_i_ka = line_flow_limit/v_max_i/1000
    max_i_ka = line_flow_limit/v_max_i

    return R,X,C,line_flow_limit,max_i_ka

def get_delaunay(all_buses,all_lines,voltage_level):
    bus_list = get_buses_at_voltage(all_buses,voltage_level)
    latlong_coord = np.zeros((len(bus_list),2))
    for i in range(len(bus_list)):
        latlong_coord[i,0] = all_buses[bus_list[i]].longitude
        latlong_coord[i,1] = all_buses[bus_list[i]].latitude 

    pca_matrix = latlong_coord 
    #Implementing the Delaunay traingulation algorithm for transmission lines and adding to the delauny_lines array
    transmissions = Delaunay(pca_matrix)
    delauny_lines = []
    for i in range(len(transmissions.simplices)):  
        delauny_lines.append([transmissions.simplices[i][0], transmissions.simplices[i][1]])
        delauny_lines.append([transmissions.simplices[i][1], transmissions.simplices[i][2]])
        delauny_lines.append([transmissions.simplices[i][0], transmissions.simplices[i][2]])

    dcount = 0
    for i in get_lines_at_voltage(all_lines,voltage_level):
        f_bus = all_lines[i].from_bus
        t_bus = all_lines[i].to_bus
        i = bus_list.index(f_bus)
        j = bus_list.index(t_bus)
        exists = False
        for idx in range(len(delauny_lines)):
            if i == delauny_lines[idx][0] and j == delauny_lines[idx][1]:
                exists = True 
            if i == delauny_lines[idx][1] and j == delauny_lines[idx][0]:
                exists = True 
        if exists:
            dcount += 1 
    return dcount 
    

def validate_lines(all_buses,all_lines,voltage_level):
    #Get buses at this voltage level
    bus_list = get_buses_at_voltage(all_buses,voltage_level)
    #Adding the Minimum Spanning Tree Lines to the all_lines array
    csr_matrix = np.zeros((len(bus_list), len(bus_list)))
    for i in range(len(bus_list)):
        for j in range(len(bus_list)):
            if(bus_list[i] < bus_list[j]):
                dist = get_distance(all_buses[bus_list[i]],all_buses[bus_list[j]])
                csr_matrix[i][j] = dist 

    mst = minimum_spanning_tree(csr_matrix).toarray()

    num_mst = 0
    dist_mst = 0
    for i in range(mst.shape[0]):
        for j in range(mst.shape[1]):
            if(mst[i][j] != 0) and (i < j):
                num_mst += 1
                dist_mst += csr_matrix[i][j]

    dist_all = 0 
    for i in range(len(all_lines)):
        if all_lines[i].voltage == voltage_level:
            dist_all += all_lines[i].length 

    return num_mst/len(get_lines_at_voltage(all_lines,voltage_level)), dist_all/dist_mst 

def create_line_net(all_buses,all_lines,all_loads,all_generators,all_transformers):
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
    return net 


def rank_lines(all_buses,delauny_lines,net,bus_list,voltage_level,line_parameters):
    d_ranking = np.zeros(len(delauny_lines))
    for idx in range(len(delauny_lines)):
        f_bus = bus_list[delauny_lines[idx][0]]
        t_bus = bus_list[delauny_lines[idx][1]]
        line_voltage = voltage_level 
        params = line_parameters[line_voltage]
        line_length = get_distance(all_buses[f_bus],all_buses[t_bus])
        R,X,C,line_flow_limit,max_i_ka = calc_line_parameters(params,line_voltage,line_length)
        Pex = 1/X*(net.res_bus['va_degree'][f_bus] - net.res_bus['va_degree'][t_bus])
        d_ranking[idx] = -0.5*np.absolute(Pex) + 2*line_length
    return d_ranking     

def find_substation_line(all_lines,all_buses,f_bus,t_bus):
    found = -1
    ft = {all_buses[f_bus].substation, all_buses[t_bus].substation}
    for i in range(len(all_lines)):
        l = all_lines[i]
        if ft == {all_buses[l.from_bus].substation, all_buses[l.to_bus].substation}:
            found = i
    return found 

def calc_penalty(all_lines,all_buses,bus_list,graph,n_components,p1,p2,p3,cl,map_lines,voltage_level):
    penalty = 0
    cl = list(cl)
    f_bus = cl[0]
    t_bus = cl[1]
    cl = set(cl)
    line_length = get_distance(all_buses[f_bus],all_buses[t_bus])
    penalty += line_length*p1*((0.2-0.01)*np.random.rand()+0.01)
    if line_length > 180 and voltage_level == 110:
        penalty += 50 
    if line_length > 230 and voltage_level == 220:
        penalty += 50 
    if line_length > 350 and voltage_level == 220:
        penalty += 50
    if line_length > 430 and voltage_level == 750:
        penalty += 50 
    # if line_length > 2*line_thresh:
    #     penalty += p1*2
    # elif line_length > line_thresh:
    #     penalty += p1 
    i = bus_list.index(f_bus)
    j = bus_list.index(t_bus)
    if j < i:
        temp = i
        i = j
        j = temp 
    graph[i,j] = 1
    new_n_components, new_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    graph[i,j] = 0
    penalty += (p2)*(new_n_components-n_components)
    if find_substation_line(all_lines,all_buses,f_bus,t_bus) != -1:
        penalty += p3
    if cl in map_lines:
        penalty -= p2*((0.15-0.05)*np.random.rand()+0.05)
        #print("in map: ", all_buses[f_bus].voltage, cl)
    return penalty 

def add_lines_ranked(all_buses,all_lines,thermal_limits,voltage_level,line_parameters,map_lines):
    #Get buses at this voltage level
    bus_list = get_buses_at_voltage(all_buses,voltage_level)
    #Adding the Minimum Spanning Tree Lines to the all_lines array
    csr_matrix = np.zeros((len(bus_list), len(bus_list)))
    for i in range(len(bus_list)):
        for j in range(len(bus_list)):
            if(bus_list[i] < bus_list[j]):
                dist = get_distance(all_buses[bus_list[i]],all_buses[bus_list[j]])
                csr_matrix[i][j] = dist 

    mst = minimum_spanning_tree(csr_matrix).toarray()

    candidate_lines = []
    for i in range(len(bus_list)):
        for j in range(len(bus_list)):
            if i < j and mst[i,j] != 0:
                candidate_lines.append({bus_list[i],bus_list[j]})

    for i in range(len(map_lines)):
        if map_lines[i] not in candidate_lines:
            candidate_lines.append(map_lines[i])

    latlong_coord = np.zeros((len(bus_list),2))
    for i in range(len(bus_list)):
        latlong_coord[i,0] = all_buses[bus_list[i]].longitude
        latlong_coord[i,1] = all_buses[bus_list[i]].latitude 

    pca_matrix = latlong_coord 
    #Implementing the Delaunay traingulation algorithm for transmission lines and adding to the delauny_lines array
    transmissions = Delaunay(pca_matrix)
    delauny_lines = []
    for i in range(len(transmissions.simplices)):  
        delauny_lines.append([transmissions.simplices[i][0], transmissions.simplices[i][1]])
        delauny_lines.append([transmissions.simplices[i][1], transmissions.simplices[i][2]])
        delauny_lines.append([transmissions.simplices[i][0], transmissions.simplices[i][2]])
        
    for i in range(len(delauny_lines)):
        f_bus = bus_list[delauny_lines[i][0]]
        t_bus = bus_list[delauny_lines[i][1]]
        if {f_bus,t_bus} not in candidate_lines:
            candidate_lines.append({f_bus,t_bus})

    print(len(candidate_lines))

    from scipy.sparse.csgraph import connected_components
    N = len(bus_list)
    orig_graph = np.zeros((N,N))
    graph = lil_matrix(orig_graph)#csr_matrix(orig_graph)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    p1 = 1
    p2 = 50
    p3 = 12
    #Add lines until connected
    nl = 0
    from_mst = 0
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    rankings = np.zeros(len(candidate_lines))
    for i in range(len(candidate_lines)):
        cl = candidate_lines[i]
        rankings[i] = calc_penalty(all_lines,all_buses,bus_list,graph,n_components,p1,p2,p3,cl,map_lines,voltage_level)
    
    when_update = 1 
    if voltage_level == 110 or voltage_level == 220:
        when_update = 20

    while(nl <= int(sub_line_ratios[voltage_level] * len(bus_list)) and len(candidate_lines) > 0):
        if nl % when_update == 0:
            for i in range(len(candidate_lines)):
                cl = candidate_lines[i]
                rankings[i] = calc_penalty(all_lines,all_buses,bus_list,graph,n_components,p1,p2,p3,cl,map_lines,voltage_level)
        best_idx = np.argmin(rankings)
        cl = candidate_lines[best_idx]
        f_bus = list(cl)[0]
        t_bus = list(cl)[1]
        i = bus_list.index(f_bus)
        j = bus_list.index(t_bus)
        if mst[i,j] != 0 or mst[j,i] != 0:
            from_mst += 1
            #print("From MST: ", nl)
        
        line_voltage = voltage_level
        line_length = get_distance(all_buses[f_bus],all_buses[t_bus])
        params = line_parameters[line_voltage]
        R,X,C,line_flow_limit,max_i_ka = calc_line_parameters(params,line_voltage,line_length)
        new_line = line.Line(voltage=line_voltage,
                            from_bus = f_bus,
                            to_bus = t_bus,
                            resistance=R,
                            reactance=X,
                            capacitance=C,
                            angle_max=ANGLE_MAX,
                            line_flow_limit=line_flow_limit,
                            length=line_length,
                            max_i_ka=max_i_ka)
        all_lines.append(new_line)
        thermal_limits.append(line_flow_limit)
        
        if j < i:
            temp = i
            i = j
            j = temp 
        graph[i,j] = 1
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        nl += 1

        if voltage_level == 750:
            print(voltage_level, ":", rankings[best_idx])
            print(cl)
            print("N components: ", n_components)
            print("Number of lines: ", nl)
            print(len(all_lines))
            print("Len candidates: ", len(candidate_lines))
            print()

        candidate_lines.remove(cl)
        rankings = np.delete(rankings,best_idx)

    return all_lines,thermal_limits 

def add_lines(all_buses,all_lines,thermal_limits,voltage_level,line_parameters):
    #Get buses at this voltage level
    bus_list = get_buses_at_voltage(all_buses,voltage_level)
    #Adding the Minimum Spanning Tree Lines to the all_lines array
    csr_matrix = np.zeros((len(bus_list), len(bus_list)))
    for i in range(len(bus_list)):
        for j in range(len(bus_list)):
            if(bus_list[i] < bus_list[j]):
                dist = get_distance(all_buses[bus_list[i]],all_buses[bus_list[j]])
                csr_matrix[i][j] = dist 

    mst = minimum_spanning_tree(csr_matrix).toarray()

    for i in range(mst.shape[0]):
        for j in range(mst.shape[1]):
            if(mst[i][j] != 0):
                line_voltage = voltage_level
                if find_line(all_lines,bus_list[i],bus_list[j],line_voltage) == -1:
                    f_bus = bus_list[i]
                    t_bus = bus_list[j]
                    line_length = get_distance(all_buses[f_bus],all_buses[t_bus])
                    params = line_parameters[line_voltage]
                    R,X,C,line_flow_limit,max_i_ka = calc_line_parameters(params,line_voltage,line_length)
                    new_line = line.Line(voltage=line_voltage,
                                        from_bus = f_bus,
                                        to_bus = t_bus,
                                        resistance=R,
                                        reactance=X,
                                        capacitance=C,
                                        angle_max=ANGLE_MAX,
                                        line_flow_limit=line_flow_limit,
                                        length=line_length,
                                        max_i_ka=max_i_ka)
                    all_lines.append(new_line)
                    thermal_limits.append(line_flow_limit)

    latlong_coord = np.zeros((len(bus_list),2))
    for i in range(len(bus_list)):
        latlong_coord[i,0] = all_buses[bus_list[i]].longitude
        latlong_coord[i,1] = all_buses[bus_list[i]].latitude 

    pca_matrix = latlong_coord 
    #Implementing the Delaunay traingulation algorithm for transmission lines and adding to the delauny_lines array
    transmissions = Delaunay(pca_matrix)
    delauny_lines = []
    for i in range(len(transmissions.simplices)):  
        delauny_lines.append([transmissions.simplices[i][0], transmissions.simplices[i][1]])
        delauny_lines.append([transmissions.simplices[i][1], transmissions.simplices[i][2]])
        delauny_lines.append([transmissions.simplices[i][0], transmissions.simplices[i][2]])

    #Adding delauny_lines to all_lines until all_lines length is equal to 1.22 * n
    dc = 0
    while(len(get_lines_at_voltage(all_lines,voltage_level)) <= int(sub_line_ratios[voltage_level] * len(bus_list)) and dc < len(delauny_lines)):
        idx = np.random.randint(0,len(delauny_lines))
        f_bus = bus_list[delauny_lines[idx][0]]
        t_bus = bus_list[delauny_lines[idx][1]]
        if find_line_all(all_lines,f_bus,t_bus) == -1: 
            line_length = get_distance(all_buses[f_bus],all_buses[t_bus])
            if line_length < LINE_MAX:
                params = line_parameters[line_voltage]
                R,X,C,line_flow_limit,max_i_ka = calc_line_parameters(params,line_voltage,line_length)
                new_line = line.Line(voltage=line_voltage,
                                    from_bus = f_bus,
                                    to_bus = t_bus,
                                    resistance=R,
                                    reactance=X,
                                    capacitance=C,
                                    angle_max=ANGLE_MAX,
                                    line_flow_limit=line_flow_limit,
                                    length=line_length,
                                    max_i_ka=max_i_ka)
                all_lines.append(new_line)
                thermal_limits.append(line_flow_limit)
        dc += 1
    print("Lines at ", voltage_level, ": ", len(get_lines_at_voltage(all_lines,voltage_level)))
    print("Desired lines: ", int(sub_line_ratios[voltage_level] * len(bus_list)))
    return all_lines, thermal_limits 

def add_transformers(all_substations, all_buses, thermal_limits):
    all_transformers = []
    for i in range(len(all_substations)):
        if len(all_substations[i].voltage) > 1:
            voltage_level = sorted(all_substations[i].voltage)
            for k in range(len(voltage_level)-1):
                l_bus = find_bus_at_voltage(voltage_level[k],i,all_buses)
                h_bus = find_bus_at_voltage(voltage_level[k+1],i,all_buses)
                params = transformer_parameters[voltage_level[k]][voltage_level[k+1]]
                new_transformer = transformer.Transformer(hv_bus = h_bus,
                                                          lv_bus = l_bus,
                                                          hv = voltage_level[k+1],
                                                          lv = voltage_level[k],
                                                          sn_mva = params["sn_mva"],
                                                          vk_percent = params["vk_percent"],
                                                          vkr_percent = params["vkr_percent"],
                                                          pfe_kw = params["pfe_kw"],
                                                          i0_percent = params["i0_percent"])
                all_transformers.append(new_transformer)
                thermal_limits.append(params["sn_mva"])
    return all_transformers, thermal_limits

def find_neighbors(pindex, triang):
    neighbors = list()
    for simplex in triang.simplices:
        if pindex in simplex:
            neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
            '''
            this is a one liner for if a simplex contains the point we`re interested in,
            extend the neighbors list by appending all the *other* point indices in the simplex
            '''
    #now we just have to strip out all the dulicate indices and return the neighbors list:
    return list(set(neighbors))

def find_second_neighbors(pindex, triang):
    second_neighbors = set()
    first_neighbors = find_neighbors(pindex, triang)
    for f in first_neighbors:
        f_neighbors = find_neighbors(f, triang)
        for s in f_neighbors:
            if (s not in first_neighbors) and s != pindex:
                second_neighbors.add(s)
    #now we just have to strip out all the dulicate indices and return the neighbors list:
    return list(second_neighbors)

def delaunay_percent(all_buses,all_lines,voltage_level):
    bus_list = get_buses_at_voltage(all_buses,voltage_level)
    latlong_coord = np.zeros((len(bus_list),2))
    for i in range(len(bus_list)):
        latlong_coord[i,0] = all_buses[bus_list[i]].longitude
        latlong_coord[i,1] = all_buses[bus_list[i]].latitude 

    pca_matrix = latlong_coord 
    #Implementing the Delaunay traingulation algorithm for transmission lines and adding to the delauny_lines array
    transmissions = Delaunay(pca_matrix)

    fnc = 0
    snc = 0

    line_list = get_lines_at_voltage(all_lines,voltage_level)
    for lidx in line_list:
        l = all_lines[lidx]
        f_bus = bus_list.index(l.from_bus)
        t_bus = bus_list.index(l.to_bus)
        first_neighbors = find_neighbors(f_bus,transmissions)
        if t_bus in first_neighbors:
            fnc += 1
        else:
            second_neighbors = find_second_neighbors(f_bus,transmissions)
            if t_bus in second_neighbors:
                snc += 1

    return fnc,snc 

def find_system_connectivity(all_buses,all_lines,all_transformers):
    from scipy.sparse.csgraph import connected_components
    N = len(all_buses)
    orig_graph = np.zeros((N,N))
    for i in range(len(all_lines)):
        f = all_lines[i].from_bus
        t = all_lines[i].to_bus
        orig_graph[f,t] = 1
        orig_graph[t,f] = 1
    for i in range(len(all_transformers)):
        h = all_transformers[i].hv_bus
        l = all_transformers[i].lv_bus
        orig_graph[h,l] = 1
        orig_graph[l,h] = 1
    graph = lil_matrix(orig_graph)#csr_matrix(orig_graph)
    return graph 