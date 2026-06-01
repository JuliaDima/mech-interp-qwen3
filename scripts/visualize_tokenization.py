"""Visualize how prompts are tokenized (pipe/ampersand vs parentheses)."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Test cases: parentheses (no spaces vs with spaces) vs pipe/ampersand
test_groups = [
    {
        "type": "type",
        "prompts": [
            ("Pos", "  Yes or No: wavelength 6, path diff 6, constructive:  ")
        ],
    }
]

for group in test_groups:
    print(f"\n{'='*100}")
    print(f"{group['type'].upper()}")
    print(f"{'='*100}")

    for label, prompt in group["prompts"]:
        print(f"\n{label}: {repr(prompt)}")
        print(f"Length: {len(prompt)} characters\n")

        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        tokens = [tokenizer.decode([id]) for id in ids]

        # Extract just the symbol part for analysis
        symbol_start = prompt.index(":") + 2  # After "balanced: "
        symbol_end = prompt.rindex(":")
        symbols = prompt[symbol_start:symbol_end]
        symbol_tokens = tokenizer(symbols, add_special_tokens=False).input_ids
        symbol_token_strs = [tokenizer.decode([id]) for id in symbol_tokens]

        print(f"Symbols: {repr(symbols)}")
        print(f"Symbol tokens: {symbol_token_strs}")
        print(f"Number of symbol tokens: {len(symbol_token_strs)} (vs {len(symbols)} chars)")
        print(f"Compression ratio: {len(symbols) / len(symbol_token_strs):.2f}x\n")

        # Show character-by-character with token boundaries
        char_idx = 0
        for i, (token_id, token_str) in enumerate(zip(ids, tokens)):
            print(f"  Token {i:2d}: ID={token_id:6d}  text={repr(token_str):20s}")
            char_idx += len(token_str)

        print()
