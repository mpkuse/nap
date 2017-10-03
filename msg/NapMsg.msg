time c_timestamp
time prev_timestamp
float32 goodness
float32 color_r
float32 color_g
float32 color_b

# Point matching related. Can ignore if not using 
int32 n_sparse_matches

# For 3way match. Only uses x,y. This represents point in an image 
geometry_msgs/Point32[] curr
geometry_msgs/Point32[] prev
geometry_msgs/Point32[] curr_m
time t_curr
time t_prev
time t_curr_m
