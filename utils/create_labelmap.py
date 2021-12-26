def create_labelmap(path_to_map):
    """
    Create a label_map from pbtxt
    Input:
        path to file
    Output:
        label_map : dict(idx:class)
    """
    labelmap = {}
    id = 1

    with open(path_to_map,'r') as f:
        for line in f:
            if "display_name" in line:
                tmp_dict = {}
                l = line.split(":")
                tmp_dict["id"] = id
                tmp_dict["name"] = l[-1].replace("\n","")[2:-1]
                labelmap[id] = tmp_dict
                id+=1
    return labelmap 