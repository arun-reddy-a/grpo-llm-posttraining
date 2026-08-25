"""Prompt construction and the epoch-aware prompt sampler."""

import json

import pytest

from grpo.data import SYSTEM_PROMPT, PromptSampler, Sample, build_prompt, load_jsonl


class FakeTokenizer:
    """Stands in for a tokenizer with a chat template."""

    chat_template = "present"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return body + ("<|assistant|>" if add_generation_prompt else "")


class TestBuildPrompt:
    def test_uses_the_chat_template_when_one_exists(self):
        prompt = build_prompt("2+2?", "be careful", FakeTokenizer())
        assert prompt == "<|system|>be careful<|user|>2+2?<|assistant|>"

    def test_generation_prompt_is_appended(self):
        """Without it the sequence does not end where the assistant turn starts."""
        assert build_prompt("q", None, FakeTokenizer()).endswith("<|assistant|>")

    def test_system_prompt_is_omitted_when_absent(self):
        assert "<|system|>" not in build_prompt("q", None, FakeTokenizer())

    def test_plain_fallback_without_a_tokenizer(self):
        assert build_prompt("2+2?", "sys", None) == "sys\n\n2+2?\n"
        assert build_prompt("2+2?", None, None) == "2+2?\n"

    def test_tokenizer_without_a_template_falls_back(self):
        class NoTemplate:
            chat_template = None

        assert build_prompt("q", None, NoTemplate()) == "q\n"

    def test_default_system_prompt_names_both_tags(self):
        assert "<think>" in SYSTEM_PROMPT and "<answer>" in SYSTEM_PROMPT


class TestLoadJsonl:
    @pytest.fixture
    def dataset(self, tmp_path):
        path = tmp_path / "data.jsonl"
        rows = [{"question": f"q{i}", "answer": str(i), "tag": "x"} for i in range(5)]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    def test_loads_every_row(self, dataset):
        samples = load_jsonl(dataset, system_prompt=None)
        assert len(samples) == 5
        assert samples[0].prompt == "q0\n" and samples[0].answer == "0"

    def test_extra_columns_are_kept_as_metadata(self, dataset):
        assert load_jsonl(dataset, system_prompt=None)[0].metadata == {"tag": "x"}

    def test_limit_stops_early(self, dataset):
        assert len(load_jsonl(dataset, limit=2, system_prompt=None)) == 2

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"question":"q","answer":"1"}\n\n\n', encoding="utf-8")
        assert len(load_jsonl(path, system_prompt=None)) == 1

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dataset file not found"):
            load_jsonl(tmp_path / "absent.jsonl")

    def test_malformed_json_names_the_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"question":"ok","answer":"1"}\n{not json}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
            load_jsonl(path)

    def test_missing_prompt_column_names_the_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"answer":"1"}\n', encoding="utf-8")
        with pytest.raises(KeyError, match="has no 'question' field"):
            load_jsonl(path)


class TestPromptSampler:
    @pytest.fixture
    def samples(self):
        return [Sample(prompt=f"p{i}", answer=str(i)) for i in range(10)]

    def test_batches_have_the_requested_size(self, samples):
        sampler = PromptSampler(samples, batch_size=4)
        assert all(len(sampler.next_batch()) == 4 for _ in range(5))

    def test_it_never_runs_out(self, samples):
        """GRPO trains for a step count, so the sampler must wrap, not stop."""
        sampler = PromptSampler(samples, batch_size=3)
        seen = [s.prompt for _ in range(20) for s in sampler.next_batch()]
        assert len(seen) == 60

    def test_epoch_advances_on_wraparound(self, samples):
        sampler = PromptSampler(samples, batch_size=5)
        sampler.next_batch()
        sampler.next_batch()
        assert sampler.epoch == 0
        sampler.next_batch()
        assert sampler.epoch == 1

    def test_an_epoch_covers_the_dataset_exactly_once(self, samples):
        sampler = PromptSampler(samples, batch_size=5)
        seen = [s.prompt for _ in range(2) for s in sampler.next_batch()]
        assert sorted(seen) == sorted(s.prompt for s in samples)

    def test_same_seed_gives_the_same_order(self, samples):
        a = PromptSampler(samples, batch_size=3, seed=42)
        b = PromptSampler(samples, batch_size=3, seed=42)
        assert [s.prompt for s in a.next_batch()] == [s.prompt for s in b.next_batch()]

    def test_different_seeds_diverge(self, samples):
        a = PromptSampler(samples, batch_size=10, seed=1)
        b = PromptSampler(samples, batch_size=10, seed=2)
        assert [s.prompt for s in a.next_batch()] != [s.prompt for s in b.next_batch()]

    def test_shuffle_off_preserves_dataset_order(self, samples):
        sampler = PromptSampler(samples, batch_size=4, shuffle=False)
        assert [s.prompt for s in sampler.next_batch()] == ["p0", "p1", "p2", "p3"]

    def test_iterator_protocol(self, samples):
        sampler = PromptSampler(samples, batch_size=2)
        # strict=False is required: the sampler is deliberately infinite.
        batches = [b for b, _ in zip(sampler, range(3), strict=False)]
        assert len(batches) == 3 and all(len(b) == 2 for b in batches)

    def test_rejects_an_empty_dataset(self):
        with pytest.raises(ValueError, match="non-empty"):
            PromptSampler([], batch_size=1)

    def test_rejects_a_batch_larger_than_the_dataset(self, samples):
        with pytest.raises(ValueError, match="exceeds dataset size"):
            PromptSampler(samples, batch_size=11)

    def test_rejects_a_non_positive_batch_size(self, samples):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            PromptSampler(samples, batch_size=0)
