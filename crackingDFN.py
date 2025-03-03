import cv2
import matplotlib.pyplot as plt
import networkx as nx

# Load your image (e.g., the skeleton image) using OpenCV
img = cv2.imread("skeleton.png")
# Convert BGR (OpenCV default) to RGB for correct display in matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
print(H, W)

# Display the image and let the user click on the nodes
# plt.figure(figsize=(8, 8))
# plt.imshow(img_rgb)
# plt.title("Click on node locations, then press Enter to finish")
# # ginput(n=0) allows an unlimited number of clicks until you press Enter
# points = plt.ginput(n=0, timeout=0)
# plt.close()

# # Build the node positions dictionary (node id mapped to (x,y) position)
# node_positions = {i: (int(x), int(y)) for i, (x, y) in enumerate(points)}
# print("Registered node positions:", node_positions)

# # Optionally, create a NetworkX graph and add nodes with these positions
# G = nx.Graph()
# G.add_nodes_from(node_positions.keys())
# for node, pos in node_positions.items():
#     G.nodes[node]["pos"] = pos

# # Visualize the graph using the registered positions
# nx.draw(
#     G, pos=node_positions, with_labels=True, node_color="lightblue", edge_color="gray"
# )
# plt.title("Graph with Manually Registered Node Positions")
# plt.show()
