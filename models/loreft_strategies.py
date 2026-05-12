import torch

def full_finetuning(gpt_layers, hidden_states, extended_attention_mask):
    """
    Standard encoder pass with no LoReFT intervention.
    Passes hidden states through every GPT layer sequentially and returns
    the final hidden states. Used for full fine-tuning as a baseline.
    """
    for layer_module in gpt_layers:
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)
    return hidden_states

def last_two_layers(gpt_layers, hidden_states, extended_attention_mask, loreft):
    """
    Applies LoReFT to ALL token positions in the last 2 layers of the GPT stack.
    For every other layer, hidden states pass through unchanged. This is a broad
    intervention, every token in the sequence is edited, not just a subset.
    No attention mask is used, so padding tokens are edited alongside real tokens.
    """
    for i, layer_module in enumerate(gpt_layers):
        hidden_states = layer_module(hidden_states, extended_attention_mask)
        if i >= len(gpt_layers) - 2:
            hidden_states = loreft(hidden_states)
    return hidden_states

def last_four_pos_n_layers(gpt_layers, hidden_states, extended_attention_mask, loreft, num_layers: int = 2):
    """
    Applies LoReFT to the last 4 token positions by raw index in the last
    `num_layers` layers of the GPT stack. The window [-4:] is a fixed positional
    slice, so it is padding-blind, if sequences are right-padded, this may edit
    padding tokens instead of real content. Useful as a simple positional baseline
    but prefer attention-mask-aware variants for padded batches.
    """
    for i, layer_module in enumerate(gpt_layers):
        hidden_states = layer_module(hidden_states, extended_attention_mask)
        if i >= len(gpt_layers) - num_layers:
            hidden_states[:, -4:, :] = loreft(hidden_states[:, -4:, :])
    return hidden_states

def last_n_nonpad_all_layers(gpt_layers, hidden_states, extended_attention_mask, loreft, attention_mask, s: int = 4):
    """
    Applies LoReFT to the last `s` non-padding tokens in every layer using a
    per-example loop. The attention mask is used to find each example's true last
    token, so padding is handled correctly. A delta tensor accumulates the edits
    and is added to hidden states after each layer, keeping the intervention
    additive rather than a hard overwrite. The per-example loop makes this simpler
    to read but slower to train than the batched equivalent below.
    """
    # Computed once, attention_mask doesn't change between layers
    last_idx = attention_mask.sum(dim=1) - 1
    for layer_module in gpt_layers:
        hidden_states = layer_module(hidden_states, extended_attention_mask)
        batch_size, seq_len, hidden_dim = hidden_states.size()

        delta = torch.zeros_like(hidden_states)
        for b in range(batch_size):
            start = max(0, last_idx[b].item() - s + 1)
            end = last_idx[b].item() + 1

            orig = hidden_states[b, start:end, :]
            edited = loreft(orig)
            delta[b, start:end, :] = edited - orig

        hidden_states = hidden_states + delta
    return hidden_states

def last_s_nonpad_all_layers_no_loop(gpt_layers, hidden_states, extended_attention_mask, loreft, attention_mask, s: int = 4):
    """
    Applies LoReFT to the last `s` non-padding tokens in every layer using
    vectorised index broadcasting, no per-example Python loop. The attention mask
    is used to locate each example's true last token. Positions created by clamping
    are masked out so they don't receive spurious edits. Mathematically equivalent
    to last_n_nonpad_all_layers but faster during training due to the fully batched
    implementation.
    """
    # Computed once, attention_mask doesn't change between layers
    last_idx = attention_mask.sum(dim=1) - 1
    for layer_module in gpt_layers:
        hidden_states = layer_module(hidden_states, extended_attention_mask)
        batch_size, seq_len, hidden_dim = hidden_states.size()

        # build indices for the last s valid positions in each example
        offsets = torch.arange(s, device=hidden_states.device).unsqueeze(0)
        pos_idx = last_idx.unsqueeze(1) - (s - 1 - offsets)
        pos_idx = pos_idx.clamp(min=0)

        # gather suffix states
        batch_idx = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)
        orig = hidden_states[batch_idx, pos_idx]

        # mask out positions that were only created by clamping
        valid_mask = (pos_idx <= last_idx.unsqueeze(1)).unsqueeze(-1).to(hidden_states.dtype)

        # apply LoReFT in one batched operation
        edited = loreft(orig)

        # keep updates on truly valid suffix positions
        delta_suffix = (edited - orig) * valid_mask

        # scatter back into a full delta tensor
        delta = torch.zeros_like(hidden_states)
        delta[batch_idx, pos_idx] += delta_suffix

        hidden_states = hidden_states + delta
    return hidden_states

def p_first_s_last_nonpad_all_layers_no_loop(gpt_layers, hidden_states, extended_attention_mask, prefix_loreft, suffix_loreft, attention_mask, p: int = 4,  s: int = 4):
    """
    Applies separate LoReFT modules to the first `p` tokens (prefix) and the last
    `s` non-padding tokens (suffix) in every layer, using vectorised indexing. Both
    interventions accumulate into a single delta tensor and are applied additively.
    This does not protect against overlap, if a sequence is shorter than p+s,
    the prefix and suffix windows can collide and their deltas will sum at shared
    positions. For overlap-safe behaviour use p_first_s_last_nonpad_all_layers_no_loop_fix.
    """
    # Computed once, attention_mask doesn't change between layers
    last_idx = attention_mask.sum(dim=1) - 1
    for layer_module in gpt_layers:
        hidden_states = layer_module(hidden_states, extended_attention_mask)
        batch_size, seq_len, hidden_dim = hidden_states.size()

        delta = torch.zeros_like(hidden_states)

        # prefix intervention
        actual_p = min(p, seq_len)
        prefix_pos = torch.arange(actual_p, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        batch_idx_prefix = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)

        prefix_orig = hidden_states[batch_idx_prefix, prefix_pos]
        prefix_edited = prefix_loreft(prefix_orig)
        delta[batch_idx_prefix, prefix_pos] += (prefix_edited - prefix_orig)

        # suffix intervention
        offsets = torch.arange(s, device=hidden_states.device).unsqueeze(0)
        suffix_pos = last_idx.unsqueeze(1) - (s - 1 - offsets)
        suffix_pos = suffix_pos.clamp(min=0)

        batch_idx_suffix = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)
        suffix_orig = hidden_states[batch_idx_suffix, suffix_pos]

        # keep only truly valid suffix positions
        valid_suffix_mask = (suffix_pos <= last_idx.unsqueeze(1)).unsqueeze(-1).to(hidden_states.dtype)

        suffix_edited = suffix_loreft(suffix_orig)
        delta[batch_idx_suffix, suffix_pos] += (suffix_edited - suffix_orig) * valid_suffix_mask

        hidden_states = hidden_states + delta
    return hidden_states

def p_first_s_last_nonpad_all_layers_no_loop_fix(gpt_layers, hidden_states, extended_attention_mask, prefix_loreft, suffix_loreft, attention_mask, p: int = 4,  s: int = 4):
    """
    Overlap-safe prefix + suffix LoReFT applied in every layer. Follows the LoReFT
    paper's specification: when a sequence is too short to fit both windows without
    overlap, the prefix gets at most half the sequence length and the suffix gets at
    most the other half, keeping them strictly disjoint. Per-example validity masks
    ensure that examples with different sequence lengths in the same batch each get
    the correct effective window size. Both modules write into a shared delta tensor
    that is applied additively at the end of each layer.
    """
    # Computed once, attention_mask doesn't change between layers
    lengths = attention_mask.sum(dim=1)
    last_idx = lengths - 1
    for layer_module in gpt_layers:
        hidden_states = layer_module(hidden_states, extended_attention_mask)
        batch_size, seq_len, hidden_dim = hidden_states.size()

        delta = torch.zeros_like(hidden_states)

        # Per paper: if n < p+s, shrink so prefix/suffix stay disjoint
        prefix_len = torch.minimum(torch.full_like(lengths, p), lengths // 2)
        suffix_len = torch.minimum(torch.full_like(lengths, s), (lengths + 1) // 2)

        batch_idx = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)

        # prefix
        max_p = int(prefix_len.max().item())
        if max_p > 0:
            prefix_offsets = torch.arange(max_p, device=hidden_states.device).unsqueeze(0)
            prefix_pos = prefix_offsets.expand(batch_size, -1)
            prefix_valid = (prefix_offsets < prefix_len.unsqueeze(1)).unsqueeze(-1).to(hidden_states.dtype)

            prefix_orig = hidden_states[batch_idx, prefix_pos]
            prefix_edited = prefix_loreft(prefix_orig)

            delta[batch_idx, prefix_pos] += (prefix_edited - prefix_orig) * prefix_valid

        # suffix
        max_s = int(suffix_len.max().item())
        if max_s > 0:
            suffix_offsets = torch.arange(max_s, device=hidden_states.device).unsqueeze(0)
            suffix_pos = last_idx.unsqueeze(1) - (max_s - 1 - suffix_offsets)
            suffix_pos = suffix_pos.clamp(min=0)

            suffix_valid = (suffix_offsets >= (max_s - suffix_len).unsqueeze(1)).unsqueeze(-1).to(hidden_states.dtype)

            suffix_orig = hidden_states[batch_idx, suffix_pos]
            suffix_edited = suffix_loreft(suffix_orig)

            delta[batch_idx, suffix_pos] += (suffix_edited - suffix_orig) * suffix_valid

        hidden_states = hidden_states + delta

    return hidden_states

