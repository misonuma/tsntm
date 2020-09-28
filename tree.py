def get_tree_idxs(tree):
    tree_idxs = {}
    tree_idxs[0] = [i for i in range(1, tree//10 +1)]
    for parent_idx in tree_idxs[0]:
        tree_idxs[parent_idx] = [parent_idx*10+i for i in range(1, tree % 10 +1)]
    return tree_idxs

def get_topic_idxs(tree_idxs):
    return [0] + [idx for child_idxs in tree_idxs.values() for idx in child_idxs]

def get_child_to_parent_idxs(tree_idxs):
    return {child_idx: parent_idx for parent_idx, child_idxs in tree_idxs.items() for child_idx in child_idxs}

def get_depth(tree_idxs, parent_idx=0, tree_depth=None, depth=1):
    if tree_depth is None: tree_depth={0: depth}

    child_idxs = tree_idxs[parent_idx]
    depth +=1
    for child_idx in child_idxs:
        tree_depth[child_idx] = depth
        if child_idx in tree_idxs: get_depth(tree_idxs, child_idx, tree_depth, depth)
    return tree_depth

def get_ancestor_idxs(leaf_idx, child_to_parent_idxs, ancestor_idxs = None):
    if ancestor_idxs is None: ancestor_idxs = [leaf_idx]
    parent_idx = child_to_parent_idxs[leaf_idx]
    ancestor_idxs += [parent_idx]
    if parent_idx in child_to_parent_idxs: get_ancestor_idxs(parent_idx, child_to_parent_idxs, ancestor_idxs)
    return ancestor_idxs[::-1]

def get_descendant_idxs(model, parent_idx, descendant_idxs = None):
    if descendant_idxs is None: descendant_idxs = [parent_idx]

    if parent_idx in model.tree_idxs:
        child_idxs = model.tree_idxs[parent_idx]
        descendant_idxs += child_idxs
        for child_idx in child_idxs:
            if child_idx in model.tree_idxs: get_descendant_idxs(model, child_idx, descendant_idxs)
    return descendant_idxs