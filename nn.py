import tensorflow as tf

class DoublyRNNCell:
    def __init__(self, dim_hidden, output_layer=None):
        self.dim_hidden = dim_hidden
        
        self.ancestral_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='ancestral')
        self.fraternal_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='fraternal')
        self.hidden_layer = tf.layers.Dense(units=dim_hidden, name='hidden')
        
        self.output_layer=output_layer
        
    def __call__(self, state_ancestral, state_fraternal, reuse=True):
        with tf.variable_scope('input', reuse=reuse):
            state_ancestral = self.ancestral_layer(state_ancestral)
            state_fraternal = self.fraternal_layer(state_fraternal)

        with tf.variable_scope('output', reuse=reuse):
            state_hidden = self.hidden_layer(state_ancestral + state_fraternal)
            if self.output_layer is not None: 
                output = self.output_layer(state_hidden)
            else:
                output = state_hidden
            
        return output, state_hidden
    
    def get_initial_state(self, name):
        initial_state = tf.get_variable(name, [1, self.dim_hidden], dtype=tf.float32)
        return initial_state
    
    def get_zero_state(self, name):
        zero_state = tf.zeros([1, self.dim_hidden], dtype=tf.float32, name=name)
        return zero_state    
    
def doubly_rnn(dim_hidden, tree_idxs, initial_state_parent=None, initial_state_sibling=None, output_layer=None, name=''):
    outputs, states_parent = {}, {}
    
    with tf.variable_scope(name, reuse=False):
        doubly_rnn_cell = DoublyRNNCell(dim_hidden, output_layer)

        if initial_state_parent is None: 
            initial_state_parent = doubly_rnn_cell.get_initial_state('init_state_parent')
        if initial_state_sibling is None: 
            initial_state_sibling = doubly_rnn_cell.get_zero_state('init_state_sibling')
        output, state_sibling = doubly_rnn_cell(initial_state_parent, initial_state_sibling, reuse=False)
        outputs[0], states_parent[0] = output, state_sibling

        for parent_idx, child_idxs in tree_idxs.items():
            state_parent = states_parent[parent_idx]
            state_sibling = initial_state_sibling
            for child_idx in child_idxs:
                output, state_sibling = doubly_rnn_cell(state_parent, state_sibling)
                outputs[child_idx], states_parent[child_idx] = output, state_sibling

    return outputs, states_parent    

class RNNCell:
    def __init__(self, dim_hidden, output_layer=None):
        self.dim_hidden = dim_hidden
        self.hidden_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='hidden')
        self.output_layer=output_layer
        
    def __call__(self, state, reuse=True):
        with tf.variable_scope('output', reuse=reuse):
            state_hidden = self.hidden_layer(state)
            if self.output_layer is not None: 
                output = self.output_layer(state_hidden)
            else:
                output = state_hidden
            
        return output, state_hidden
    
    def get_initial_state(self, name):
        initial_state = tf.get_variable(name, [1, self.dim_hidden], dtype=tf.float32)
        return initial_state
    
    def get_zero_state(self, name):
        zero_state = tf.zeros([1, self.dim_hidden], dtype=tf.float32, name=name)
        return zero_state
    
def rnn(dim_hidden, max_depth, initial_state=None, output_layer=None, name='', concat=True):
    outputs, states_hidden = [], []
    
    with tf.variable_scope(name, reuse=False):
        rnn_cell = RNNCell(dim_hidden, output_layer)
        
        if initial_state is not None: 
            state_hidden = initial_state
        else:
            state_hidden = rnn_cell.get_initial_state('init_state')
        
        for depth in range(max_depth):
            if depth == 0:                
                output, state_hidden = rnn_cell(state_hidden, reuse=False)
            else:
                output, state_hidden = rnn_cell(state_hidden, reuse=True)
            outputs.append(output)
            states_hidden.append(state_hidden)

    outputs = tf.concat(outputs, 1) if concat else tf.concat(outputs, 0)
    states_hidden = tf.concat(states_hidden, 0)
    return outputs, states_hidden

def tsbp(tree_sticks_topic, tree_idxs):
    tree_prob_topic = {}
    tree_prob_leaf = {}
    # calculate topic probability and save
    tree_prob_topic[0] = 1.

    for parent_idx, child_idxs in tree_idxs.items():
        rest_prob_topic = tree_prob_topic[parent_idx]
        for child_idx in child_idxs:
            stick_topic = tree_sticks_topic[child_idx]
            if child_idx == child_idxs[-1]:
                prob_topic = rest_prob_topic * 1.
            else:
                prob_topic = rest_prob_topic * stick_topic

            if not child_idx in tree_idxs: # leaf childs
                tree_prob_leaf[child_idx] = prob_topic
            else:
                tree_prob_topic[child_idx] = prob_topic

            rest_prob_topic -= prob_topic
    return tree_prob_leaf

def sbp(sticks_depth, max_depth):
    prob_depth_list = []
    rest_prob_depth = 1.
    for depth in range(max_depth):
        stick_depth = tf.expand_dims(sticks_depth[:, depth], 1)
        if depth == max_depth -1:
            prob_depth = rest_prob_depth * 1.
        else:
            prob_depth = rest_prob_depth * stick_depth
        prob_depth_list.append(prob_depth)
        rest_prob_depth -= prob_depth

    prob_depth = tf.concat(prob_depth_list, 1)
    return prob_depth

def nhdp(tree_sticks_path, tree_sticks_depth, tree_idxs):
    tree_prob_path = {}
    tree_rest_prob_depth = {}
    tree_prob_topic = {}
    # calculate topic probability and save
    tree_prob_path[0] = 1.
    tree_rest_prob_depth[0] = 1. - tree_sticks_depth[0]
    tree_prob_topic[0] = tree_prob_path[0] * tree_sticks_depth[0]

    for parent_idx, child_idxs in tree_idxs.items():
        rest_prob_path = tree_prob_path[parent_idx]
        for child_idx in child_idxs:
            stick_path = tree_sticks_path[child_idx]
            if child_idx == child_idxs[-1]:
                prob_path = rest_prob_path * 1.
            else:
                prob_path = rest_prob_path * stick_path

            tree_prob_path[child_idx] = prob_path
            rest_prob_path -= prob_path
            
            if not child_idx in tree_idxs: # leaf childs
                tree_prob_topic[child_idx] = tree_prob_path[child_idx] * tree_rest_prob_depth[parent_idx] * 1.
            else:
                tree_prob_topic[child_idx] = tree_prob_path[child_idx] * tree_rest_prob_depth[parent_idx] * tree_sticks_depth[child_idx]
                tree_rest_prob_depth[child_idx] = tree_rest_prob_depth[parent_idx] * (1-tree_sticks_depth[child_idx])
            
    return tree_prob_topic