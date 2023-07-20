from numpy import allclose
from rayuela.base.semiring import Tropical, Real
from rayuela.cfg.parser import Parser, EarleyParser
from rayuela.cfg.cfg import CFG
from rayuela.base.misc import symify

pickles_path = "autograding_tests/pickles"
hw_path = pickles_path + "/hw6"

def test_earley_example_tropical():
    
    R = Tropical

    cfg = CFG.from_string(
        """
        A → b:	0.142
        A → c:	0.24
        A → a:	0.448
        A → d:	0.33
        A → A A:	0.14
        A → A B:	0.398
        A → A D:	0.301
        A → B A:	0.079
        A → B C:	0.403
        A → D A:	0.136
        A → D C:	0.389
        A → C A:	0.284
        A → C D:	0.112
        A → C C:	0.125
        B → b:	0.39
        B → c:	0.433
        B → a:	0.435
        B → d:	0.359
        B → A C:	0.121
        B → B B:	0.036
        B → D B:	0.476
        B → D C:	0.352
        B → C A:	0.307
        B → C B:	0.044
        C → b:	0.391
        C → c:	0.173
        C → a:	0.324
        C → d:	0.456
        C → A B:	0.401
        C → A D:	0.357
        C → B B:	0.064
        C → B D:	0.062
        C → C D:	0.099
        D → b:	0.103
        D → c:	0.035
        D → a:	0.297
        D → d:	0.15
        D → A A:	0.361
        D → A D:	0.33
        D → B A:	0.04
        D → B D:	0.268
        D → B C:	0.123
        D → C A:	0.313
        D → C B:	0.274
        S → b:	0.463
        S → c:	0.046
        S → a:	0.456
        S → d:	0.169
        S → A A:	0.207
        S → A C:	0.104
        S → B A:	0.349
        S → B C:	0.451
        S → D D:	0.15
        S → C A:	0.199
        S → C B:	0.497
        S → C C:	0.287
        """.strip(), R)

    input = symify("abc")
    ep = EarleyParser(cfg)
    parser = Parser(cfg)

    assert (allclose(float(ep.earley(input)), float(parser.cky(input)), 1e-6))

def test_earley_example_real():
    
    R = Real

    cfg = CFG.from_string(
        """
        A → b:	0.142
        A → c:	0.24
        A → a:	0.448
        A → d:	0.33
        A → A A:	0.14
        A → A B:	0.398
        A → A D:	0.301
        A → B A:	0.079
        A → B C:	0.403
        A → D A:	0.136
        A → D C:	0.389
        A → C A:	0.284
        A → C D:	0.112
        A → C C:	0.125
        B → b:	0.39
        B → c:	0.433
        B → a:	0.435
        B → d:	0.359
        B → A C:	0.121
        B → B B:	0.036
        B → D B:	0.476
        B → D C:	0.352
        B → C A:	0.307
        B → C B:	0.044
        C → b:	0.391
        C → c:	0.173
        C → a:	0.324
        C → d:	0.456
        C → A B:	0.401
        C → A D:	0.357
        C → B B:	0.064
        C → B D:	0.062
        C → C D:	0.099
        D → b:	0.103
        D → c:	0.035
        D → a:	0.297
        D → d:	0.15
        D → A A:	0.361
        D → A D:	0.33
        D → B A:	0.04
        D → B D:	0.268
        D → B C:	0.123
        D → C A:	0.313
        D → C B:	0.274
        S → b:	0.463
        S → c:	0.046
        S → a:	0.456
        S → d:	0.169
        S → A A:	0.207
        S → A C:	0.104
        S → B A:	0.349
        S → B C:	0.451
        S → D D:	0.15
        S → C A:	0.199
        S → C B:	0.497
        S → C C:	0.287
        """.strip(), R)

    input = symify("abc")
    ep = EarleyParser(cfg)
    parser = Parser(cfg)

    assert (allclose(float(ep.earley(input)), float(parser.cky(input)), 1e-6))