from _constants import BLOCKED, UNBLOCKED, UNKNOWN
from itertools import product
from _constants import COLOR
num_of_blockages = -1
block_dict = {}
blockages = -1

def initBlockagesDictionary(blockages):
    global block_dict
    global  num_of_blockages
    for i, blockage in enumerate(blockages):
        block_dict[blockage] = i
    num_of_blockages = len(block_dict)

class State:
    def __init__(self, vertex, blockages_belief, graph):
        self.vertex = vertex
        self.blockages_belief = blockages_belief
        self.curUtility = -float('inf')
        self.curBestAction = None
        self.graph = graph
        self.belief_states = None
        self.consecutive_states = None

    def vertexNearUnknownBlockage(self):
        for edge, idx in block_dict.items():
            if self.blockages_belief[idx] == UNKNOWN and edge.occurs_in_vertex_element(self.vertex):
                return True
        return False

    def isEdgeBlocked(self, action):
        if action in block_dict:
            return self.blockages_belief[block_dict[action]] == BLOCKED
        return False

    def getUnknownRelevantBlockages(self, edge):
        unknown_blockages = []
        other_vertex = edge.get_other_end(self.vertex)
        adjacent_edges = self.graph.get_adjacent_edges(other_vertex)
        for edge in adjacent_edges:
            if self.isEdgeUnknown(edge):
                unknown_blockages.append(edge)
        return unknown_blockages

    def isEdgeUnknown(self, edge):
        if not (edge in block_dict):
            return False
        return self.blockages_belief[block_dict[edge]] == UNKNOWN

    def setConsecutiveStates(self, belief_states):
        self.belief_states = belief_states
        self.consecutive_states = []
        for action in self.graph.get_adjacent_edges(self.vertex):
            cosecutive_states_of_action = []
            if self.isEdgeBlocked(action):
                continue
            unknown_relevant_blockages = self.getUnknownRelevantBlockages(action)
            for unknown_relevant_blockages_belief in (list(tup) for tup in (product([BLOCKED, UNBLOCKED], repeat=len(unknown_relevant_blockages)))):
                new_state, prob_new_state = self.getStateAndProb(unknown_relevant_blockages, unknown_relevant_blockages_belief, self.belief_states, action.get_other_end(self.vertex))
                cosecutive_states_of_action.append([new_state, prob_new_state])

            self.consecutive_states.append([action, cosecutive_states_of_action])

    def getStateAndProb(self, unknown_relevant_blockages, unknown_relevant_blockages_belief, belief_states, destination_vertex):
        desired_belief = []
        prob = 1
        for edge, belief in zip(block_dict, self.blockages_belief):
            if belief != UNKNOWN:
                desired_belief.append(belief)
            elif edge not in unknown_relevant_blockages:
                desired_belief.append(UNKNOWN)
            else:
                consecutive_belief = unknown_relevant_blockages_belief[
                    unknown_relevant_blockages.index(edge)
                ]
                desired_belief.append(consecutive_belief)
                if consecutive_belief == BLOCKED:
                    prob *= edge.block_prob
                else:
                    prob *= (1-edge.block_prob)
        for state in belief_states:
            if state.blockages_belief == desired_belief and state.vertex == destination_vertex:
                return state, prob
        raise Exception("No state found")


    def unreachableState(self):
        adjacent_edges = self.graph.get_adjacent_edges(self.vertex)
        for edge in adjacent_edges:
            if edge.noRisk():
                return False
            if self.blockages_belief[block_dict[edge]] in [UNBLOCKED, UNKNOWN]:
                return False
        return True

    @classmethod
    def addToStr(cls, msg, tabs, newline=True):
        pr = ("\t"*tabs)+msg
        return pr+"\n" if newline else pr

    def toString(self, is_target, t=0, action_string="best move to", with_utility=True):
        addToStr = State.addToStr
        pr=""
        pr += addToStr(COLOR['UNDERLINE'] + COLOR['BOLD'] + "State" + COLOR['END'], t)
        t += 1
        pr += addToStr(COLOR['UNDERLINE'] + "Vertex" + COLOR['END'] + ": " + COLOR['BLUE'] + COLOR['BOLD'] + str(
            self.vertex.key) + COLOR['END'], t)
        pr += addToStr(COLOR['UNDERLINE'] + "Blockages" + COLOR['END'] + ":", t)
        t += 1
        for edge, belief in zip(list(block_dict.keys()), self.blockages_belief):
            pr += addToStr("edge {} {}".format(edge.verticesToString(), belief), t)
        t -= 1
        if with_utility:
            pr += addToStr(COLOR['UNDERLINE'] + "utility" + COLOR['END'] + ": "
                       + COLOR['RED'] + str(self.curUtility) + COLOR['END'],
                       t)
        if is_target:
            pr += addToStr(COLOR['BOLD'] + COLOR['GREEN'] + "is target" + COLOR['END'], t)
        else:
            pr += addToStr(
                COLOR['UNDERLINE'] + action_string + COLOR['END'] + ": " + COLOR['BOLD'] + COLOR['GREEN'] + str(
                    self.curBestAction.get_other_end(self.vertex).key) + COLOR['END'], t)
        t -= 1
        return pr