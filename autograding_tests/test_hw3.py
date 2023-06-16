from rayuela.base.misc import compare_fsas, is_topologically_sorted_scc
from rayuela.fsa.fsa import FSA
from rayuela.fsa.fst import FST
from rayuela.base.semiring import Tropical, Real
from rayuela.fsa.state import PairState, State
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.scc import SCC
from rayuela.fsa.pathsum import Pathsum, Strategy
import pickle
import numpy as np

pickles_path = "autograding_tests/pickles"
hw_path = pickles_path + "/hw3"

with open(f"{pickles_path}/fsas.pkl", 'rb') as f:
    fsas = pickle.load(f)

fsa = FSA(R=Real)

fsa.add_arc(State(2), Sym('a'), State(8), Real(0.1))

fsa.add_arc(State(3), Sym('c'), State(5), Real(0.4))
fsa.add_arc(State(3), Sym('a'), State(7), Real(0.7))

fsa.add_arc(State(5), Sym('a'), State(2), Real(0.3))
fsa.add_arc(State(5), Sym('a'), State(7), Real(0.8))

fsa.set_I(State(3), Real(0.3))
fsa.set_F(State(2), Real(0.4))
fsa.set_F(State(7), Real(0.2))



def test_kosajaru_example_1():

    components = [frozenset({State(3)}),
                frozenset({State(5)}),
                frozenset({State(7)}),
                frozenset({State(2)}),
                frozenset({State(8)})]
    sccs = SCC(fsa)

    gt = components
    computed = list(sccs.scc())

    # All elements are present
    assert all([elem in gt for elem in computed]) and all([elem in computed for elem in gt])
    # Computed components are topologically sorted
    assert is_topologically_sorted_scc(computed, fsa)

def test_kosajaru_example_2():
    F = FSA(R=Real)
    F.set_I(State(0), F.R(1 / 2))

    F.add_arc(State(0), Sym('a'), State(1), F.R(1 / 2.0))
    F.add_arc(State(0), Sym('a'), State(2), F.R(1 / 5.0))

    F.add_arc(State(1), Sym('b'), State(1), F.R(1 / 8.0))
    F.add_arc(State(1), Sym('c'), State(3), F.R(1 / 10.0))

    F.add_arc(State(2), Sym('d'), State(3), F.R(1 / 5.0))

    F.add_arc(State(3), Sym('b'), State(2), F.R(1 / 10.0))

    F.add_F(State(3), F.R(1 / 4))

    components = [frozenset({State(0)}), frozenset({State(1)}), frozenset({State(2), State(3)})]

    gt = components
    computed = list(SCC(F).scc())
    
    # All elements are present
    assert all([elem in gt for elem in computed]) and all([elem in computed for elem in gt])
    # Computed components are topologically sorted. As fsa is not acyclic, there is no topological order.
    # assert is_topologically_sorted_scc(computed, F)

def test_kosajaru_example_3():
    F = FSA(R=Real)
    F.set_I(State(0), F.R(1 / 2))

    F.add_arc(State(0), Sym('a'), State(1), F.R(1 / 2.0))
    F.add_arc(State(0), Sym('a'), State(2), F.R(1 / 5.0))

    F.add_arc(State(1), Sym('c'), State(3), F.R(1 / 10.0))
    F.add_arc(State(1), Sym('c'), State(4), F.R(1 / 10.0))

    F.add_arc(State(2), Sym('d'), State(3), F.R(1 / 5.0))
    F.add_arc(State(2), Sym('c'), State(5), F.R(1 / 10.0))

    F.add_arc(State(3), Sym('b'), State(2), F.R(1 / 10.0))

    F.add_arc(State(4), Sym('c'), State(1), F.R(1 / 10.0))

    F.add_arc(State(5), Sym('c'), State(3), F.R(1 / 10.0))

    F.add_F(State(3), F.R(1 / 4))

    components = [frozenset({State(0)}), frozenset({State(1), State(4)}), frozenset({State(2), State(3), State(5)})]

    gt = components
    computed = list(SCC(F).scc())
    
    # All elements are present
    assert all([elem in gt for elem in computed]) and all([elem in computed for elem in gt])
    # Computed components are topologically sorted. As fsa is not acyclic, there is no topological order.
    # assert is_topologically_sorted_scc(computed, F)
    
    
def test_kosajaru():
    with open(f"{hw_path}/sccs.pkl", 'rb') as f:
        sccs = pickle.load(f)
    for fsa, scc in zip(fsas, sccs):
        
        # Topological order only exists for acyclic fsas. 
        if not fsa.acyclic:
            continue
        sccs_fsa = SCC(fsa)

        gt = scc
        computed = list(sccs_fsa.scc())
        

        # All elements are present
        assert all([elem in gt for elem in computed]) and all([elem in computed for elem in gt])

        # Computed components are topologically sorted. Only check if there are more than 1 SCC (there's no orden in a list of only one element)
        if len(computed) > 1:
            assert is_topologically_sorted_scc(computed, fsa)

        


def test_decomposed_lehman_example():
   
    assert np.allclose(float(Pathsum(fsa).pathsum(strategy=Strategy.DECOMPOSED_LEHMANN)),float(Real(0.0756)), atol=1e-3)

def test_decomposed_lehman():
    with open(f"{hw_path}/pathsums.pkl", 'rb') as f:
        pathsums = pickle.load(f)
    for fsa, pathsum in zip(fsas, pathsums):
        assert np.allclose(float(Pathsum(fsa).pathsum(strategy=Strategy.DECOMPOSED_LEHMANN)),float(pathsum), atol=1e-3)

def test_top_composition_example():
    
    
    fst1 = FST(Real)
    fst1.add_arc(State(0), 'c','a', State(1), Real(0.336))
    fst1.add_arc(State(0), 'a','a', State(1), Real(0.187))
    fst1.add_arc(State(0), 'c','c', State(1), Real(0.132))
    fst1.set_I(State(0), Real(0.685))
    fst1.add_arc(State(1), 'a','a', State(0), Real(0.459))
    fst1.add_F(State(1), Real(0.393))
    fst2 = FST(Real)
    fst2.add_arc(State(0), 'c','c', State(0), Real(0.47))
    fst2.add_arc(State(0), 'a','a', State(1), Real(0.463))
    fst2.set_I(State(0), Real(0.664))
    fst2.add_arc(State(1), 'c','c', State(0), Real(0.467))
    fst2.add_arc(State(1), 'c','a', State(1), Real(0.251))
    fst2.add_F(State(1), Real(0.09))
    TOP = FST(Real)
    TOP.add_arc(PairState(0, 1), 'a','a', PairState(1, 0), Real(0.212517))
    TOP.add_arc(PairState(1, 0), 'c','a', PairState(0, 1), Real(0.156912))
    TOP.add_arc(PairState(1, 0), 'c','a', PairState(1, 1), Real(0.046937))
    TOP.add_arc(PairState(1, 0), 'c','c', PairState(0, 1), Real(0.061644))
    TOP.add_arc(PairState(1, 1), 'c','a', PairState(1, 0), Real(0.115209))
    TOP.add_F(PairState(1,1), Real(0.03537))
    TOP.add_arc(PairState(0, 0), 'c','a', PairState(0, 1), Real(0.15792))
    TOP.add_arc(PairState(0, 0), 'c','c', PairState(0, 1), Real(0.06204))
    TOP.add_arc(PairState(0, 0), 'a','a', PairState(1, 1), Real(0.086581))
    TOP.set_I(PairState(0,0), Real(0.45484))

    top = fst1.top_compose(fst2).trim()
    
    # TODO: Not sure if this is the best assertion. Specially interested in checking the labels
    # assert top.trim().__str__() == TOP.trim().__str__()
    # for q1, q2 in zip(top.Q, TOP.Q):
    #     assert list(top.arcs(q1)) == list(TOP.arcs(q2))
    assert compare_fsas(TOP, top)

def test_composition():

    with open(f"{hw_path}/top_composition.pkl", 'rb') as f:
        top_composition = pickle.load(f)
    with open(f"{hw_path}/bottom_composition.pkl", 'rb') as f:
        bottom_composition = pickle.load(f)

    try:
        top_compose_works = all([compare_fsas(top, left.top_compose(right)) for top, left, right in top_composition])
    except:
        top_compose_works = False
    try:
        bottom_compose_works = all([compare_fsas(top, left.bottom_compose(right)) for top, left, right in bottom_composition])
    except: 
        bottom_compose_works = False

    assert top_compose_works or bottom_compose_works
