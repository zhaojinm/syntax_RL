from syntax_rl.rewards.ccg import score_ccg
from syntax_rl.rewards.ptb import score_constituency


def test_ccg_exact_and_partial_scores():
    gold = "{N Cats} {S[dcl]\\NP sleep} {. .}"
    assert score_ccg(gold, gold) == 1.0
    assert score_ccg("{NP Cats} {S[dcl]\\NP sleep} {. .}", gold) == 2 / 3


def test_ccg_invalid_alignment_is_zero():
    gold = "{N Cats} {S[dcl]\\NP sleep} {. .}"
    assert score_ccg("{N Dogs} {S[dcl]\\NP sleep} {. .}", gold) == 0.0
    assert score_ccg("{N Cats}", gold) == 0.0


def test_answer_tags_are_optional():
    gold = "{N Cats} {S[dcl]\\NP sleep} {. .}"
    assert score_ccg(f"<answer>{gold}</answer>", gold) == 1.0


def test_ptb_identical_tree_scores_100():
    tree = "(TOP (S (NP (NNS Cats)) (VP (VBP sleep)) (. .)))"
    assert score_constituency(tree, tree) == 100.0

