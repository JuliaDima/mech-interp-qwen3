"""Tests for prompt building functionality."""

from __future__ import annotations

from mechinterp_qwen3.build_prompts import (
    generate_examples,
    make_gt_prompt,
)


class TestMakeGTPrompt:
    """Test the make_gt_prompt function."""

    def test_format_structure(self):
        """Test that prompt has expected structure."""
        prompt, expected = make_gt_prompt(42, 17)

        assert "You are solving a simple comparison task" in prompt
        assert "A = 42" in prompt
        assert "B = 17" in prompt
        assert "Answer: " in prompt
        assert prompt.endswith("Answer: ")

    def test_expected_answer_a_larger(self):
        """Test expected answer when A > B."""
        prompt, expected = make_gt_prompt(100, 50)
        assert expected == "A"

    def test_expected_answer_b_larger(self):
        """Test expected answer when B > A."""
        prompt, expected = make_gt_prompt(25, 75)
        assert expected == "B"

    def test_expected_answer_edge_case(self):
        """Test edge cases with very close numbers."""
        prompt, expected = make_gt_prompt(100, 99)
        assert expected == "A"

        prompt, expected = make_gt_prompt(99, 100)
        assert expected == "B"

    def test_prompt_consistency(self):
        """Test that same inputs produce same outputs."""
        prompt1, expected1 = make_gt_prompt(42, 17)
        prompt2, expected2 = make_gt_prompt(42, 17)

        assert prompt1 == prompt2
        assert expected1 == expected2


class TestGenerateExamples:
    """Test the generate_examples function."""

    def test_correct_count(self):
        """Test that correct number of examples are generated."""
        n = 10
        examples = generate_examples(n=n, seed=0, low=0, high=999)
        assert len(examples) == n

    def test_determinism_same_seed(self):
        """Test that same seed produces same examples."""
        examples1 = generate_examples(n=20, seed=42, low=0, high=999)
        examples2 = generate_examples(n=20, seed=42, low=0, high=999)

        assert len(examples1) == len(examples2)
        for ex1, ex2 in zip(examples1, examples2, strict=False):
            assert ex1.a == ex2.a
            assert ex1.b == ex2.b
            assert ex1.prompt == ex2.prompt
            assert ex1.expected == ex2.expected

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different examples."""
        examples1 = generate_examples(n=20, seed=0, low=0, high=999)
        examples2 = generate_examples(n=20, seed=1, low=0, high=999)

        # At least some examples should differ
        differences = sum(
            1
            for ex1, ex2 in zip(examples1, examples2, strict=False)
            if ex1.a != ex2.a or ex1.b != ex2.b
        )
        assert differences > 0

    def test_no_ties(self):
        """Test that a != b for all examples."""
        examples = generate_examples(n=50, seed=0, low=0, high=999)

        for ex in examples:
            assert ex.a != ex.b, f"Found tie: a={ex.a}, b={ex.b}"

    def test_values_in_range(self):
        """Test that generated values are within specified range."""
        low, high = 10, 100
        examples = generate_examples(n=30, seed=0, low=low, high=high)

        for ex in examples:
            assert low <= ex.a <= high
            assert low <= ex.b <= high

    def test_prompt_id_format(self):
        """Test that prompt IDs are correctly formatted."""
        examples = generate_examples(n=5, seed=0, low=0, high=999)

        for i, ex in enumerate(examples):
            expected_id = f"gt_{i:04d}"
            assert ex.prompt_id == expected_id

    def test_expected_answer_correctness(self):
        """Test that expected answers match the comparison logic."""
        examples = generate_examples(n=30, seed=0, low=0, high=999)

        for ex in examples:
            if ex.a > ex.b:
                assert ex.expected == "A"
            else:
                assert ex.expected == "B"

    def test_example_dataclass_fields(self):
        """Test that GTExample has all required fields."""
        examples = generate_examples(n=1, seed=0, low=0, high=999)
        ex = examples[0]

        assert hasattr(ex, "prompt_id")
        assert hasattr(ex, "a")
        assert hasattr(ex, "b")
        assert hasattr(ex, "prompt")
        assert hasattr(ex, "expected")

        assert isinstance(ex.prompt_id, str)
        assert isinstance(ex.a, int)
        assert isinstance(ex.b, int)
        assert isinstance(ex.prompt, str)
        assert isinstance(ex.expected, str)
