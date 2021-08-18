import numpy.random as random
from Graph import Graph
from Vertex import Vertex
from Edge import Edge
from state import State
import state as state_file
from itertools import product
import _settings
from _constants import COLOR
from _constants import THERE_ARE_PEOPLE, VERTICES, EDGES, PATH, MOST_LIKELY_PATH
from _constants import VERTEX, EDGE, RESET_EVIDENCE, ADD_EVIDENCE, PROB_REASONING
from _constants import QUIT, SHOW_GRAPH, SHOW_BAYES_NET, MENU
from _constants import PRINT_POLICY, STOCHASTIC_EXAMPLE, USER_INPUT_EXAMPLE
from _constants import BLOCKED, UNBLOCKED, UNKNOWN
from _constants import TARGET

def _setInputFunction(func):
    global input
    input = func

class HurricaneEnv:
    def __init__(self, graph_input_file_name):
        self.number_of_vertices = None
        self.read_graph_from_file(graph_input_file_name)

    def read_graph_from_file(self, file_name):
        file = open(file_name, "r")
        vertices = []
        edges = []
        for line in file:
            words = line.split()
            if ';' in words:
                i=words.index(';')
                words = words[0:i]
            if not words:  # if the list is empty
                continue
            first_word = words[0]
            if first_word[0] == ';':
                continue
            if first_word[1] == 'N':
                self.number_of_vertices = int(words[1])
                continue
            if first_word[1] == 'D':
                self.deadline = float(words[1])
                continue
            if first_word[1] == 'V':
                key = int(first_word[2:])
                if len(words) <= 2:
                    evac_prob = 0.0
                else:
                    third_word = words[2]
                    evac_prob = float(third_word)
                vertex = Vertex(key, evac_prob)
                vertices.append(vertex)
                continue
            if first_word[1] == 'E':
                key = int(first_word[2:])
                second_word = words[1]
                third_word = words[2]
                fourth_word = words[3][1:]
                if len(words)>4:
                    fifth_word = words[4][1:]
                    block_prob = float(fifth_word)
                else:
                    block_prob = 0
                vertex1 = Vertex.get_vertex(vertices, int(second_word))
                vertex2 = Vertex.get_vertex(vertices, int(third_word))
                weight = int(fourth_word)
                edge = Edge(vertex1, vertex2, weight, key, block_prob)
                edges.append(edge)
                continue
            if first_word == '#Start' or first_word == '#start':
                second_word = words[1]
                start = int(second_word)
                continue
            if first_word == '#Target' or first_word == '#target':
                second_word = words[1]
                target = int(second_word)
                continue
        self.graph = Graph(vertices, edges)
        self.start = self.graph.get_vertex(start)
        self.target = self.graph.get_vertex(target)

    def run_env(self):
        self.printInitialData()
        blockages = self.graph.getBlockages()
        state_file.initBlockagesDictionary(blockages)
        self.constructStates(blockages)
        self.initializeStates()
        self.valueIteration()
        while True:
            user_action = self.getUserAction()
            if user_action == QUIT:
                return
            # SHOW_GRAPH
            elif user_action == PRINT_POLICY:
                self.printPolicy()
            elif user_action == STOCHASTIC_EXAMPLE:
                self.runStochasticExample()
            elif user_action == USER_INPUT_EXAMPLE:
                self.runUserInputExample()
            elif user_action == SHOW_GRAPH:
                self.printInitialData()
            else:
                raise Exception("Not a valid option")

    def getUserAction(self):
        bolding = COLOR['UNDERLINE']+COLOR['BOLD']
        print("What action to commit?\n"
              +bolding+"P"+COLOR['END']+"rint Policy/"
              +"run "+bolding+"S"+COLOR['END']+"tochastic example/"
              +"run "+bolding+"U"+COLOR['END']+"ser input example/"
              +bolding+"G"+COLOR['END']+"raph/"
              +bolding+"Q"+COLOR['END']+"uit"
              +"\t")

        choice = {
            '1' : PRINT_POLICY,
            'P' : PRINT_POLICY,
            'p' : PRINT_POLICY,
            '2': STOCHASTIC_EXAMPLE,
            'S': STOCHASTIC_EXAMPLE,
            's': STOCHASTIC_EXAMPLE,
            '3': USER_INPUT_EXAMPLE,
            'U': USER_INPUT_EXAMPLE,
            'u': USER_INPUT_EXAMPLE,
            '4': SHOW_GRAPH,
            'G': SHOW_GRAPH,
            'g': SHOW_GRAPH,
            '5': QUIT,
            'Q': QUIT,
            'q': QUIT
        }.get(input())

        return choice

    def printInitialData(self):
        print('graph: \n'+self.graph.to_print(start=self.start, target=self.target))

    def constructStates(self, blockages):
        self.belief_states = []
        for vertex in self.graph.vertices:
            for blockages_belief in (list(tup) for tup in product([BLOCKED, UNBLOCKED, UNKNOWN], repeat=len(blockages))):
                state = State(vertex, blockages_belief, self.graph)
                self.belief_states.append(state)
        states_to_delete = []
        for state in self.belief_states:
            if state.vertexNearUnknownBlockage():
                states_to_delete.append(state)
            elif state.unreachableState():
                states_to_delete.append(state)
        for state in states_to_delete:
            self.belief_states.remove(state)
        for state in self.belief_states:
            state.setConsecutiveStates(self.belief_states)
        return

    def initializeStates(self):
        for state in self.belief_states:
            state.curUtility = 0.0
            if self.isTarget(state):
                state.curBestAction = TARGET
            else:
                state.curBestAction = self.graph.get_adjacent_edges(state.vertex)[0]

    def valueIteration(self):
        print_steps = HurricaneEnv.yesNoToBool(input("Do you want to print each step? (y/n)\t"))
        changed = True
        iteration = 0
        while changed:
            if iteration>10000:
                raise Exception("didn't converge")
            iteration += 1
            if print_steps:
                self.printAtEachStep(iteration)
            changed = False
            for state in self.belief_states:
                if self.isTarget(state):
                    continue
                best_action_util = -float('inf')
                best_action = None
                for [action, consecutive_states] in state.consecutive_states:
                    action_util = self.utilityFromAction(action, consecutive_states)
                    if action_util > best_action_util:
                        best_action_util = action_util
                        best_action = action
                if state.curUtility != best_action_util or state.curBestAction != best_action:
                    changed = True
                    state.curUtility = best_action_util
                    state.curBestAction = best_action

    @classmethod
    def addToStr(cls, msg, tabs, newline=True):
        pr = ("\t"*tabs)+msg
        return pr+"\n" if newline else pr

    def printPolicy(self, t=0):
        addToStr = HurricaneEnv.addToStr
        pr = ""
        pr += addToStr("Policy:",t)
        t += 1
        for state in self.belief_states:
            if state.curBestAction is None:
                continue
            pr += state.toString(is_target=self.isTarget(state), t=t)
        print(pr)

    def utilityFromAction(self, edge, consecutive_states):
        util = 0
        for state, prob_for_state in consecutive_states:
            util += prob_for_state * state.curUtility
        return float(util - edge.weight)

    def isTarget(self, state):
        return self.target == state.vertex

    def printAtEachStep(self, iteration):
        print(COLOR['BOLD']+COLOR['UNDERLINE']+
              COLOR['PURPLE']+"Iteration "+str(iteration)+COLOR['END'])
        self.printPolicy(t=1)

    @classmethod
    def yesNoToBool(cls, inp):
        if inp == "y" or inp == "Y":
            return True
        if inp == "n" or inp == "N":
            return False
        raise Exception("Invalid input")

    def showPath(self, blockages_data):
        state = self.getNextStateUserInputSetting(blockages_data)
        pr = "Run environment in a user input settings:\n"
        while not self.isTarget(state):
            pr += state.toString(is_target=False, t=1, action_string="Moving to", with_utility=False)
            action = state.curBestAction
            state = self.getNextStateUserInputSetting(blockages_data, state, action)
        pr += state.toString(is_target=True, t=1, action_string="Moving to", with_utility=False)
        print(pr)

    def runStochasticExample(self):
        state = self.getFirstStateStochasticSettings()
        pr = "Run environment in a stochastic graph:\n"
        while not self.isTarget(state):
            pr += state.toString(is_target=False, t=1, action_string="Moving to", with_utility=False)
            action = state.curBestAction
            consec_states_and_probs = next(x[1] for x in state.consecutive_states if x[0]==action)
            consec_states = list(x[0] for x in consec_states_and_probs)
            consec_probs = list(x[1] for x in consec_states_and_probs)
            state = random.choice(consec_states, p=consec_probs)
        pr += state.toString(is_target=True, t=1, action_string="Moving to", with_utility=False)
        print(pr)
        pr = "The graph that was constructed:\n"
        pr += self.stochasticBlockagesToString(state.blockages_belief)
        print(pr)

    def getFirstStateStochasticSettings(self):
        unknown_blockages = state_file.block_dict.keys()
        adjacent_edges = self.graph.get_adjacent_edges(self.start)
        blockages_data = []
        for edge in unknown_blockages:
            if edge in adjacent_edges:
                block_prob = edge.block_prob
                blockages_data.append(random.choice([BLOCKED, UNBLOCKED], p=[block_prob, 1-block_prob]))
            else:
                blockages_data.append(UNKNOWN)
        for state in self.belief_states:
            if self.start == state.vertex:
                if blockages_data == state.blockages_belief:
                    return state
        raise Exception("No first state found")

    def stochasticBlockagesToString(self, belief_blockages):
        pr = ""
        for idx, blockage in enumerate(state_file.block_dict.keys()):
            pr += blockage.to_print()
            belief = belief_blockages[idx]
            pr += "\t, "+COLOR['DARKCYAN']+COLOR['BOLD']+belief.upper()+COLOR['END']
            if belief == UNKNOWN:
                prob = float(blockage.block_prob)
                draw = random.choice([BLOCKED, UNBLOCKED], p=[prob, 1-prob])
                pr+= "\tDrawn to be: "+COLOR['DARKCYAN']+draw.upper()+COLOR['END']
            pr += "\n"
        return pr

    def runUserInputExample(self):
        blockages_data = []
        for blockage in state_file.block_dict.keys():
            print(blockage.to_print())
            inp = input("\t\tIs blocked? (y/n)\t")
            if inp == 'y' or inp == 'Y':
                blockages_data.append(BLOCKED)
            elif inp == 'n' or inp == 'N':
                blockages_data.append(UNBLOCKED)
            else:
                raise Exception("Invalid choice")
        self.showPath(blockages_data)

    def getNextStateUserInputSetting(self, blockages_data, prev_state=None, action=None):
        desired_blockages_belief = []
        if prev_state is None:
            desired_vertex = self.start
            prev_blockages_belief = [UNKNOWN]*state_file.num_of_blockages
            consec_states = list(state for state in self.belief_states if state.vertex==desired_vertex)
        else:
            desired_vertex = action.get_other_end(prev_state.vertex)
            prev_blockages_belief = prev_state.blockages_belief
            consec_states_and_probs = next(x[1] for x in prev_state.consecutive_states if x[0] == action)
            consec_states = list(x[0] for x in consec_states_and_probs)
        adjacent_edges = self.graph.get_adjacent_edges(desired_vertex)
        for edge, idx in state_file.block_dict.items():
            if edge not in adjacent_edges:
                desired_blockages_belief.append(prev_blockages_belief[idx])
            else:
                cur_blockage_belief = prev_state.blockages_belief[idx]
                if cur_blockage_belief != UNKNOWN:
                    desired_blockages_belief.append(cur_blockage_belief)
                else:
                    desired_blockages_belief.append(blockages_data[idx])
        for state in consec_states:
            if state.blockages_belief == desired_blockages_belief:
                return state
        raise Exception("No state found")





