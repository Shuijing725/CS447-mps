from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxz")
E = set("e")
U = set("u")
NPTR = set("nptr")
PT = set("pt")

# Implement your solution here
def buildFST():
    # print("Your task is to implement a better FST in the buildFST() function, using the methods described here")
    # print("You may define additional methods in this module (hw1_fst.py) as desired")
    #
    # The states (you need to add more)
    # ---------------------------------------
    # 
    f = FST("q0") # q0 is the initial (non-accepting) state
    # f.addState("q1") # a non-accepting state
    f.addState("q_ing") # a non-accepting state
    f.addState("q_EOW", True) # an accepting state (you shouldn't need any additional accepting states)
    
    f.addState("q1.1")
    f.addState("q1.2")
    f.addState("q2.1.1") # single vow state for a, i, o, u
    f.addState("q2.1.2") # single vow state for e
    f.addState("q2.2") # multi vow state
    f.addState("q3.1")
    f.addState("q3.2")

    #
    # The transitions (you need to add more):
    # ---------------------------------------
    # transduce every element in this set to itself: 
    # f.addSetTransition("q0", AZ, "q1")
    # AZ-E =  the set AZ without the elements in the set E
    # f.addSetTransition("q1", AZ-E, "q1")

    # get rid of this transition! (it overgenerates):
    # f.addSetTransition("q1", AZ, "q_ing")

    # map the empty string to ing: 
    f.addTransition("q_ing", "", "ing", "q_EOW")


    # rule 1
    f.addSetTransition("q0", CONS, "q1.1")
    f.addTransition("q1.1", "e", "", "q_ing")
    f.addSetTransition("q1.1", AZ - VOWS, "q0")
    f.addSetTransition("q1.1", VOWS - E, "q2.1.1")
    # if the word did not end with "e"
    f.addTransition("q1.1", "e", "e", "q1.2")
    f.addTransition("q1.1", "", "", "q_ing")
    f.addSetTransition("q1.2", AZ, "q0")

    # rule 2
    f.addSetTransition("q0", VOWS - E, "q2.1.1")
    # q2.1.1: single vow state (except e)
    f.addSetTransition("q2.1.1", CONS, "q1.1")
    # double n, p, t, r
    f.addTransition("q2.1.1", "n", "nn", "q_ing")
    f.addTransition("q2.1.1", "p", "pp", "q_ing")
    f.addTransition("q2.1.1", "t", "tt", "q_ing")
    f.addTransition("q2.1.1", "r", "rr", "q_ing")
    # f.addTransition("q2.1.1", "e", "", "q_ing")
    # single -> multiple vows
    f.addSetTransition("q2.1.1", VOWS - E, "q2.2")
    # q2.2: multi vow state
    f.addSetTransition("q2.2", VOWS, "q2.2")
    f.addSetTransition("q2.2", CONS, "q0")
    f.addTransition("q2.2", "", "", "q_ing")
    # q2.1.2:
    f.addSetTransition("q0", E, "q2.1.2")
    f.addTransition("q2.1.2", "p", "pp", "q_ing")
    f.addTransition("q2.1.2", "t", "tt", "q_ing")
    f.addSetTransition("q2.1.2", VOWS, "q2.2")
    f.addSetTransition("q2.1.2", CONS - PT, "q0")


    # rule 3
    f.addTransition("q0", "i", "", "q3.1")
    f.addTransition("q3.1", "e", "y", "q_ing")
    # f.addSetTransition("q3.1", AZ-E, "q0")

    # rest of them
    f.addTransition("q0", "", "", "q_ing")
    # f.addSetTransition("q0", CONS | E, "q0")

    # Return your completed FST
    return f
    

if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)
