def expected_shape(max_node_num, as_image,flatten_adjacency, full_adjacency):
    number_of_upper_entries = int(max_node_num*(max_node_num-1.)*.5)
    if flatten_adjacency:
        if full_adjacency:
            D = max_node_num * max_node_num
            if as_image:
                shape = [1, 1, D]
            else:
                shape = [D]
        else:
            D = number_of_upper_entries
            if as_image:
                shape = [1, 1, D]
            else:
                shape = [D]
    else:
        if full_adjacency:
            D = max_node_num * max_node_num
            if as_image:
                shape = [1, max_node_num, max_node_num]
            else:
                shape = [max_node_num, max_node_num]
        else:
            raise ValueError("No Flatten and No Full Adjacency incompatible for data")

    return D, shape

if __name__=="__main__":
    # Usage:
    # Replace the following variables with appropriate values
    flatten_adjacency = True  # or False
    full_adjacency = False  # or False
    as_image = True  # or False
    max_node_num = 10  # for example

    D, shape = expected_shape(max_node_num, as_image,flatten_adjacency, full_adjacency)
    print("Expected D:", D)
    print("Expected Shape:", shape)
