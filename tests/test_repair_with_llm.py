from evaluation.repair_with_llm import (
    ccg_pairs,
    clean_candidate,
    ptb_leaves,
    source_from_record,
    validate_ccg,
    validate_ptb,
    validate_ptb_historical,
)


def test_source_from_string_prompt():
    record = {
        "prompt": "system\nhelper\nuser\nGenerate the constituency tree of the given sentence.\nCats sleep .\nassistant\n"
    }
    assert source_from_record(record, "ptb") == "Cats sleep ."


def test_ptb_validation():
    tree = "(TOP (S (NP (NNS Cats)) (VP (VBP sleep)) (. .)))"
    assert ptb_leaves(tree) == ["Cats", "sleep", "."]
    validate_ptb(tree, "Cats sleep .")


def test_historical_ptb_check_does_not_repair_equal_length_lexical_errors():
    tree = "(TOP (S (NP (NNS Dogs)) (VP (VBP sleep)) (. .)))"
    validate_ptb_historical(tree, "Cats sleep .")


def test_ccg_validation():
    sequence = "{N Cats} {S[dcl]\\NP sleep} {. .}"
    assert ccg_pairs(sequence) == [("N", "Cats"), ("S[dcl]\\NP", "sleep"), (".", ".")]
    validate_ccg(sequence, "Cats sleep .")


def test_candidate_cleanup():
    assert clean_candidate("```text\n<answer>{N Cats}</answer>\n```") == "{N Cats}"
