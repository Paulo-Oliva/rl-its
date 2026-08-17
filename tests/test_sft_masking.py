import pytest


@pytest.mark.model
def test_only_teacher_turns_contribute_to_the_sft_loss(tokenizer):
    from its.training.sft import _tokenize

    messages = [
        {
            "role": "system",
            "content": "You are a Socratic tutor. [Problem] 2x=10 [Answer] 5"
        },
        {
            "role": "assistant",
            "content": "What operation undoes multiplying by two?"
        },
        {
            "role": "user",
            "content": "Dividing, I think."
        },
        {
            "role": "assistant",
            "content": "Good. Try it on both sides."
        },
    ]
    ex = _tokenize(tokenizer, messages, max_length=512)
    assert ex is not None

    supervised = [i for i, l in enumerate(ex.labels.tolist()) if l != -100]
    assert supervised, "some tokens must be trained on"
    text = tokenizer.decode([ex.input_ids[i] for i in supervised])
    assert "What operation undoes" in text and "Try it on both sides" in text
    assert "Socratic tutor" not in text, "the system prompt is masked"
    assert "Dividing, I think" not in text, "the student's turn is masked"
    # where a token is trained on, its label is the token itself
    for i in supervised:
        assert ex.labels[i] == ex.input_ids[i]
