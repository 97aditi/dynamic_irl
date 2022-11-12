import numpy as np

# level order traversal of binary tree marked with nodes marked as in the maze
lev0 = [0]
lev1 = [1,2]
lev2 = [3,4,6,5]
lev3 = [8,7,9,10,13,14,12,11]
lev4 = [18, 17, 15, 16, 19, 20, 22, 21, 27, 28, 30, 29, 26, 25, 23, 24]
lev5 = [37,38,36,35,32,31,33,34,40,39,41,42,45,46,44,43,56,55,57,58,61,62,60,59,53,54,52,51,48,47,49,50]
lev6 = [75,76,78,77,74,73,71,72,66,65,63,64,67,68,70,69,82,81,79,80,83,84,86,85,91,92,94,93,90,89,87,88,114,113,111,112,115,116,118,117,123,124,126,125,122,121,119,120,107,108,110,109,106,105,103,104,98,97,95,96,99,100,102,101]

levelorder_traversal_rosenberg_maze = lev0+lev1+lev2+lev3+lev4+lev5+lev6

def rosenbergnode_to_ournode(rosenberg_node):
    """ inputs an int which represents a node from Rosenberg et al and converts it to its binary search tree representation
        input:
            rosenberg_node (int): between 0--127
        returns:
            our_node (int): between 1-127
    """
    if rosenberg_node==127:
        our_node = 1
    else:
        index = levelorder_traversal_rosenberg_maze.index(rosenberg_node)
        our_node = index+1
    return our_node

def ournode_to_rosenbergnode(our_node):
    '''
    takes our node and converts back to rosenberg space
    :param our_node:
    :return:
    '''
    rosenberg_node = levelorder_traversal_rosenberg_maze[our_node-1]
    return rosenberg_node

if __name__ == "__main__":
    node = 101
    assert rosenbergnode_to_ournode(ournode_to_rosenbergnode(node)) == node
    print(rosenbergnode_to_ournode(57))