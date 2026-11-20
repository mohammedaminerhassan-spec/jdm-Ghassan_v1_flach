# Checkpoints

## Ghassan v1 Flash thinking fixes

- Outcome: Fixed thinking token alignment: chat_template now uses <think>/</think> consistent with generator and GGUF export; fixed corrupted ChatML control tokens; added tests/test_chat_template_thinking.cpp. Full CPU build and ctest pass.
- Decisions: Kept thinking opt-in and fail-closed; no silent fallback. Model tokenizer must contain <think_on>, <think>, </think> plus Granite role tags. No CUDA changes needed for this fix.
- Files: tokenizer/chat_template.cpp; tests/test_chat_template_thinking.cpp; build/ (generated)
- Next: If user needs thinking on a Granite tokenizer without the 3 tokens, add a token-alias mechanism or extend embedding vocab; otherwise done.
