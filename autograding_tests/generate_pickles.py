from typing import List, Set
from rayuela.base.semiring import Rational, Real, Semiring, Tropical, MaxPlus
from rayuela.base.symbol import Sym, ε
from rayuela.base.misc import is_pathsum_positive, filter_negative_pathsums
from rayuela.cfg.cfg import CFG
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
import dill as pickle                               #using dill instead of pickle bc pickle cannot pickle FSAs
import random
import os
import argparse
from collections import Counter
from tqdm import tqdm

from rayuela.fsa.random import random_machine
from rayuela.fsa.scc import SCC
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.fsa.transformer import Transformer

from rayuela.cfg.nonterminal import NT, S, Triplet, Delta
from rayuela.cfg.random import random_cfg, random_cfg_cnf



def config_parser():
    parser = argparse.ArgumentParser(description="Pickles generation script")
    parser.add_argument("--hw", type=int, default=0, help="The number of homework you want to generate the pickles for. --hw=0 to generate all (default)")
    return parser

def generate_fsas(pickles_path, sigma: Set[Sym] = None, n_machines: int = 1000, semiring: Semiring = Real) -> List[FSA]:

    if not sigma:
        sigma = set([Sym('a'), Sym('b'), Sym('c'), Sym('d'), Sym('e'), Sym('f')])

    print(f"Generating {n_machines} random {semiring} FSAs...")

    # Caveates about random_machine method:
    # - Control to generate deterministic, acyclic, trimmed machines by parameters
    # - Generates some machines that can be pushed, some machines that cannot be pushed. Not really sure why tho
    # - Only positive pathsum filtering: We need fsas with a positive pathsum so operations like concatenate don't produce diverged fsas
    # - Generates minimized fsas

    fsas = []
    booleans = (True, False)
    while len(fsas) < n_machines:
        acyclic = random.choice(booleans)
        deterministic = random.choice(booleans)
        num_states = random.randint(3,8)
        i = random.randint(2, len(sigma))
        fsa = random_machine(set(random.sample(list(sigma),i)), semiring, num_states=num_states, acyclic=acyclic, deterministic=deterministic) #+2 to avoid single state machines
        try: # Can throw a ZeroDivisionError when computing the pathsum with LEHMANN, in that case, skip the fst
            if is_pathsum_positive(fsa):
                fsas.append(fsa)
        except:
            continue

    with open(f"{pickles_path}/fsas.pkl", 'wb') as f:
        pickle.dump(fsas, f)
    return fsas

def deterministic_fsas(pickles_path: str, fsas: List[FSA]) -> None:

    print("Generating fsas deterministic results...")

    deterministic_results = [fsa.deterministic for fsa in fsas]
    print("Number of deterministic FSAs", Counter(deterministic_results))
    with open(f"{pickles_path}/deterministic_fsas.pkl", 'wb') as f:
        pickle.dump(fsas, f)
    with open(f"{pickles_path}/deterministic_results.pkl", 'wb') as f:
        pickle.dump(deterministic_results, f)

def pushed_fsas(pickles_path: str, fsas: List[FSA]) -> None:
    
    
    print("Generating pushed FSAs...")
    pushed_fsas = []

    # fsa.push() throw exceptions for some fsas. Unknown reason. Ideally, we push half of the fsas randomly.
    # if random.random() < 0.5:
    #     pushed_fsas.append(fsa.push())
    continue_pushing = True
    for fsa in fsas:
        if continue_pushing:
            try:
                pushed_fsas.append(fsa.push()) # Might throw exception
            except:
                pushed_fsas.append(fsa)
            if len(pushed_fsas) == len(fsas)/2:
                continue_pushing = False
        else:
            pushed_fsas.append(fsa)
            

    with open(f"{pickles_path}/pushed_fsas.pkl", 'wb') as f:
        pickle.dump(pushed_fsas, f)

    pushed_results = [fsa.pushed for fsa in pushed_fsas]
    print("Number of pushed FSAs", Counter(pushed_results))
    with open(f"{pickles_path}/pushed_results.pkl", 'wb') as f:
        pickle.dump(pushed_results, f)


def reversed_fsas(pickles_path: str, fsas: List[FSA]) -> None:

    
    reversed_fsas = [fsa.reverse() for fsa in fsas]
    # reversed_fsas = filter_negative_pathsums(reversed_fsas)

    print(f"Generating {len(reversed_fsas)} reversed FSAs...")

    with open(f"{pickles_path}/reversed_fsas.pkl", 'wb') as f:
        pickle.dump(reversed_fsas, f)


def accessible_and_coaccessible_states(pickles_path: str, fsas: List[FSA]) -> None:


    print("Generating accessible & coaccesible states...")
    accessible_states_fsas = [fsa.accessible() for fsa in fsas]
    with open(f"{pickles_path}/accessible_states.pkl", 'wb') as f:
        pickle.dump(accessible_states_fsas, f)

    coaccessible_states_fsas = [fsa.coaccessible() for fsa in fsas]
    with open(f"{pickles_path}/coaccessible_states.pkl", 'wb') as f:
        pickle.dump(coaccessible_states_fsas, f)

def unions(pickles_path: str, left_fsas: List[FSA], right_fsas: List[FSA]) -> None:
    unions = [fsa1.union(fsa2) for fsa1, fsa2 in zip(left_fsas, right_fsas)]
    # unions = filter_negative_pathsums(unions)
    print(f"Generating {len(unions)} union FSAs...")

    with open(f"{pickles_path}/union_fsas.pkl", 'wb') as f:
        pickle.dump(unions, f)

def concatenations(pickles_path: str, left_fsas: List[FSA], right_fsas: List[FSA]) -> None:
    
    concats = [fsa1.concatenate(fsa2) for fsa1, fsa2 in zip(left_fsas, right_fsas)]
    # concats = filter_negative_pathsums(concats)
    print(f"Generating {len(concats)} concatenation FSAs...")
    with open(f"{pickles_path}/concat_fsas.pkl", 'wb') as f:
        pickle.dump(concats, f)


def kleene_closure(pickles_path: str, fsas: List[FSA]) -> None:
    fsas = filter_negative_pathsums(fsas)
    kleenes = [(fsa, fsa.kleene_closure()) for fsa in fsas]
    kleenes = [(fsa, kfsa) for fsa, kfsa in kleenes if is_pathsum_positive(kfsa)]
    print(f"Generating {len(kleenes)} Kleene closure FSAs...")

    with open(f"{pickles_path}/kleene_fsas.pkl", 'wb') as f:
        pickle.dump(kleenes, f)

def viterbi_fwd(pickles_path: str, fsas: List[FSA]) -> None:
    α = [Pathsum(fsa).forward(Strategy.VITERBI) for fsa in fsas if fsa.acyclic]
    print(f"Generating {len(α)} Viterbi forward α...")

    with open(f"{pickles_path}/hw2_viterbi_fwd.pkl", 'wb') as f:
        pickle.dump(α, f)

def edge_marginals(pickles_path: str, fsas: List[FSA]) -> None:
    marginals = [fsa.edge_marginals() for fsa in fsas if fsa.acyclic]
    print(f"Generating {len(marginals)} edge merginals...")

    with open(f"{pickles_path}/hw2_marginals.pkl", 'wb') as f:
        pickle.dump(marginals, f)

def coaccessible_intersection(pickles_path: str, fsas: List[FSA]) -> None:
    left_fsas, right_fsas = fsas[:500], fsas[500:]
    co_fsas = [left.coaccessible_intersection(right) for left, right in zip(left_fsas, right_fsas)]
    with open(f"{pickles_path}/hw2_cointersec.pkl", 'wb') as f:
        pickle.dump(co_fsas, f)

def scc_decomposition(pickles_path: str, fsas: List[FSA]) -> None:
    print("Generating SCCs...")
    sccs = [list(SCC(fsa).scc()) for fsa in fsas]
    with open(f"{pickles_path}/sccs.pkl", 'wb') as f:
        pickle.dump(sccs, f)

def decomposed_lehman_pathsum(pickles_path: str, fsas: List[FSA]) -> None:
    print("Generating decomposed lehman pathsums...")
    pathsums = [Pathsum(fsa).pathsum(strategy=Strategy.DECOMPOSED_LEHMANN) for fsa in fsas]
    with open(f"{pickles_path}/pathsums.pkl", 'wb') as f:
        pickle.dump(pathsums, f)

def composition(pickles_path: str, sigma: Set[Sym] = None, n_fsts: int = 100, semiring: Semiring = Real) -> None:

    if not sigma:
        sigma = set([Sym('a'), Sym('b'), Sym('c'), Sym('d'), Sym('e'), Sym('f')])

    print(f"Generating {n_fsts} random FSTs...")

    # Caveates about random_machine method:
    # - Control to generate deterministic, acyclic, trimmed machines by parameters
    # - Generates some machines that can be pushed, some machines that cannot be pushed. Not really sure why tho
    # - Only positive pathsum filtering: We need fsas with a positive pathsum so operations like concatenate don't produce diverged fsas
    # - Generates minimized fsas

    fsts = []
    booleans = (True, False)
    while len(fsts) < n_fsts:
        acyclic = random.choice(booleans)
        deterministic = random.choice(booleans)
        num_states = random.randint(3,8)
        i = random.randint(2, len(sigma))
        fst = random_machine(set(random.sample(list(sigma), i)), semiring, num_states=num_states, acyclic=acyclic, deterministic=deterministic, fst=True, trimmed=False) 
        try:
            if is_pathsum_positive(fst): # Can throw a ZeroDivisionError when computing the pathsum with LEHMANN, in that case, skip the fst
                fsts.append(fst)
        except:
            continue
    
    middle = int(n_fsts/2)
    left, right = fsts[:middle], fsts[middle:]
    top_composition = []
    bottom_composition = []
    for l, r in zip(left, right):
        top = l.top_compose(r)
        if is_pathsum_positive(top):
            top_composition.append((top, l, r))
        bottom = l.bottom_compose(r)
        if is_pathsum_positive(top):
            bottom_composition.append((bottom, l, r))

    print(f"Generating composition. top: {len(top_composition)}. Bottom: {len(bottom_composition)}.")

    with open(f"{pickles_path}/top_composition.pkl", 'wb') as f:
        pickle.dump(top_composition, f)

    with open(f"{pickles_path}/bottom_composition.pkl", 'wb') as f:
        pickle.dump(bottom_composition, f)


def determinization(pickles_path: str, fsas: List[FSA]) -> None:
    
    determinized_fsas = []
    determinizable_fsas = []
    print("Generating determization test...")
    for idx, fsa in tqdm(enumerate(fsas[:50])):
        # Filter out already deterministic machines
        if fsa.deterministic:
            continue
        try:
            dfsa = fsa.determinize(timeout=100)
        except:
            continue
        if is_pathsum_positive(dfsa) and is_pathsum_positive(fsa):
            determinized_fsas.append(dfsa)
            determinizable_fsas.append(fsa)
    
    print(f"Generating {len(determinized_fsas)} determinized fsas...")

    with open(f"{pickles_path}/determinized_fsas.pkl", 'wb') as f:
        pickle.dump(determinized_fsas, f)

    with open(f"{pickles_path}/determinizable_fsas.pkl", 'wb') as f:
        pickle.dump(determinizable_fsas, f)

def bellmanford(pickles_path: str, fsas: List[FSA]):
    
    αs = [fsa.forward(Strategy.BELLMANFORD) for fsa in fsas]
    print(f"Generating {len(αs)} bellmanford...")

    with open(f"{pickles_path}/bellmanford_fwd.pkl", 'wb') as f:
        pickle.dump(αs, f)

def johnson(pickles_path: str, fsas: List[FSA]):
    Ws = [Pathsum(fsa).johnson() for fsa in fsas]
    print(f"Generating {len(Ws)} Johnson...")
    print(f"{sum([fsa.pushed for fsa in fsas])} of them were already pushed")
    with open(f"{pickles_path}/johnson.pkl", 'wb') as f:
        pickle.dump(Ws, f)

def minimization(pickles_path: str, fsas: List[FSA]):
    minimized_fsas = []
    for fsa in fsas:
        if fsa.deterministic: # According to the Transformer.minimize_partition, the fsa must be deterministic
            try:
                fsa.minimize(strategy="partition") # Using the other method produces KeyErrors when testing fsa.minimize(strategy="partition") in test_hw5.py
                minimized_fsas.append((fsa, Transformer.minimize_nfa(fsa)))
            except:
                continue
    print(f"Generated {len(minimized_fsas)} minimized fsas")
    with open(f"{pickles_path}/minimized_fsas.pkl", 'wb') as f:
        pickle.dump(minimized_fsas, f)

def equivalence(pickles_path: str, fsas: List[FSA]):
    
    #We need to check all fsas are determinizable to test equivalence. Hence, selecting only those able to compute equivalence.
    print("Generating equivalence test...")
    max_per_method = 10
    hw_path = "autograding_tests/pickles" + "/hw1"
    # Reversed fsas
    with open(f"{hw_path}/reversed_fsas.pkl", 'rb') as f:
        reversed_fsas = pickle.load(f)
    
    rfsas = []
    gold_rfsas = []
    for fsa, rfsa in zip(fsas, reversed_fsas):
        try:
            assert Transformer.equivalent_nfa(rfsa, fsa.reverse())
            assert len(list(rfsa.I)) == 1 # Ensure single initial state
            assert len(list(rfsa.push().I)) # Sometimes there are fsas w/o I states due to numerical inestability 
            rfsa.determinize(timeout=200) # We need determinizable fsas
            rfsas.append(fsa)
            gold_rfsas.append(rfsa)
        except:
            continue
        if len(rfsas) == max_per_method:
            break

    # Union
    with open(f"{hw_path}/union_fsas.pkl", 'rb') as f:
        union_fsas = pickle.load(f)
    
    ufsas = []
    gold_ufsas = []

    middle = int(len(fsas)/2)
    for left_fsa, right_fsa, union_fsa in zip(fsas[:middle], fsas[middle:], union_fsas):
        try:
            assert Transformer.equivalent_nfa(union_fsa, left_fsa.union(right_fsa))
            assert len(list(union_fsa.I)) == 1 # Ensure single initial state
            assert len(list(union_fsa.push().I)) # Sometimes there are fsas w/o I states due to numerical inestability 
            union_fsa.determinize(timeout=200) # We need determinizable fsas
            ufsas.append((left_fsa, right_fsa))
            gold_ufsas.append(union_fsa)
        except:
            continue
        if len(ufsas) == max_per_method:
            break

    # Concat
    with open(f"{hw_path}/concat_fsas.pkl", 'rb') as f:
        concat_fsas = pickle.load(f)
    
    cfsas = []
    gold_cfsas = []
    middle = int(len(fsas)/2)
    for left_fsa, right_fsa, concat_fsa in zip(fsas[:middle], fsas[middle:], concat_fsas):
        try:
            assert Transformer.equivalent_nfa(concat_fsa, left_fsa.concatenate(right_fsa))
            assert len(list(concat_fsa.I)) == 1 # Ensure single initial state
            assert len(list(concat_fsa.push().I)) # Sometimes there are fsas w/o I states due to numerical inestability 
            concat_fsa.determinize(timeout=200) # We need determinizable fsas
            cfsas.append((left_fsa, right_fsa))
            gold_cfsas.append(concat_fsa)
        except:
            continue
        if len(cfsas) == max_per_method:
            break

    # Kleene closure
    with open(f"{hw_path}/kleene_fsas.pkl", 'rb') as f:
        kleene_fsas = pickle.load(f)
    
    kfsas = []
    gold_kfsas = []
    for fsa, kleene in zip(fsas, kleene_fsas):
        try:
            assert Transformer.equivalent_nfa(kleene, fsa.kleene_closure())
            assert len(list(kleene.I)) == 1 # Ensure single initial state
            assert len(list(kleene.push().I)) # Sometimes there are fsas w/o I states due to numerical inestability 
            kleene.determinize(timeout=200) # We need determinizable fsas
            kfsas.append(fsa)
            gold_kfsas.append(kleene)
        except:
            continue
        if len(kfsas) == max_per_method:
            break
    
    print(f"Equivalence: Generated {len(rfsas)} reversed, {len(ufsas)} union, {len(cfsas)} concat, {len(kfsas)} kleene, ")
    equivalents = {"reverse": (rfsas, gold_rfsas),
                "union": (ufsas, gold_ufsas),
                "concat": (cfsas, gold_cfsas),
                "kleene": (kfsas, gold_kfsas)}

    with open(f"{pickles_path}/equivalence.pkl", 'wb') as f:
        pickle.dump(equivalents, f)

def generate_cfgs(pickles_path, sigma: Set[Sym] = None, terminals: Set = None, n_cfgs: int = 1000, semiring: Semiring = Real, cnf=False) -> List[CFG]:

    if not sigma:
        sigma = set([ε, Sym('a'), Sym('b'), Sym('c'), Sym('d'), Sym('e'), Sym('f')])
    
    if not terminals:
        terminals = set([S, NT("A"), NT("B"), NT("C"), NT("D")])

    print(f"Generating {n_cfgs} random CFGss...")

    cfgs = []
    while len(cfgs) < n_cfgs:
        i = random.randint(2, len(sigma))
        j = random.randint(2, len(terminals))
        if cnf:
            cfg = random_cfg_cnf(Sigma = set(random.sample(list(sigma), i)), V=set(random.sample(list(terminals), j)), R=semiring)
        else:
            cfg = random_cfg(Sigma = set(random.sample(list(sigma), i)), V=set(random.sample(list(terminals), j)), R=semiring)
        
        cfgs.append(cfg)
    if cnf:
        name = "cnfs"
    else:
        name = "cfgs"
    with open(f"{pickles_path}/{name}.pkl", 'wb') as f:
        pickle.dump(cfgs, f)
    return cfgs



if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()
    pickles_path = "autograding_tests/pickles"
    os.makedirs(pickles_path, exist_ok=True)
    print(f"Generating pickles for hw {args.hw}")
    
    fsas = generate_fsas(pickles_path, semiring=Real)

    if args.hw == 1 or not args.hw:
        pickles_hw_path = pickles_path + "/hw1"
        os.makedirs(pickles_hw_path, exist_ok=True)
        deterministic_fsas(pickles_path=pickles_hw_path, fsas=fsas)
        pushed_fsas(pickles_path=pickles_hw_path, fsas=fsas)
        reversed_fsas(pickles_path=pickles_hw_path, fsas=fsas)
        accessible_and_coaccessible_states(pickles_path=pickles_hw_path, fsas=fsas)
        kleene_closure(pickles_path=pickles_hw_path, fsas=fsas)

        middle = int(len(fsas)/2)
        unions(pickles_path=pickles_hw_path, left_fsas=fsas[:middle], right_fsas=fsas[middle:])
        concatenations(pickles_path=pickles_hw_path, left_fsas=fsas[:middle], right_fsas=fsas[middle:])

    if args.hw == 2 or not args.hw:
        pickles_hw_path = pickles_path + "/hw2"
        os.makedirs(pickles_hw_path, exist_ok=True)
        edge_marginals(pickles_path=pickles_hw_path, fsas=fsas)
        viterbi_fwd(pickles_path=pickles_hw_path, fsas=fsas)
        coaccessible_intersection(pickles_path=pickles_hw_path, fsas=fsas)

    if args.hw == 3 or not args.hw:
        pickles_hw_path = pickles_path + "/hw3"
        os.makedirs(pickles_hw_path, exist_ok=True)
        scc_decomposition(pickles_path=pickles_hw_path, fsas=fsas)
        decomposed_lehman_pathsum(pickles_path=pickles_hw_path, fsas=fsas)
        composition(pickles_path=pickles_hw_path)
    
    if args.hw == 4 or not args.hw:
        pickles_hw_path = pickles_path + "/hw4"
        os.makedirs(pickles_hw_path, exist_ok=True)
        determinization(pickles_path=pickles_hw_path, fsas=fsas)

    if args.hw == 5 or not args.hw:
        pickles_hw_path = pickles_path + "/hw5"
        os.makedirs(pickles_hw_path, exist_ok=True)
        tropical_fsas = generate_fsas(pickles_path=pickles_hw_path, n_machines=200, semiring=MaxPlus)
        bellmanford(pickles_path=pickles_hw_path, fsas=tropical_fsas)
        johnson(pickles_path=pickles_hw_path, fsas=tropical_fsas)
        minimization(pickles_path=pickles_hw_path, fsas=fsas)
        equivalence(pickles_path=pickles_hw_path, fsas=fsas)

    if args.hw == 6 or not args.hw:
        pickles_hw_path = pickles_path + "/hw6"
        os.makedirs(pickles_hw_path, exist_ok=True)
        cfgs = generate_cfgs(pickles_path=pickles_hw_path, n_cfgs=100, semiring=Tropical)
        cfg_cnfs = generate_cfgs(pickles_path=pickles_hw_path, n_cfgs=100, semiring=Tropical, cnf=True)


