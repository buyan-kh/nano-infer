export type Level = {
  requests: number;
  generated_tokens: number;
  wall_s: number;
  throughput_tps: number;
  ttft_p50_ms: number;
  ttft_p99_ms: number;
  latency_p50_ms: number;
  latency_p99_ms: number;
  concurrency: number;
};

export type BenchSystem = {
  id: string;
  label: string;
  levels: Level[];
};

export type MemorySystem = {
  requests: number;
  generated_tokens: number;
  wall_s: number;
  throughput_tps: number;
  ttft_p50_ms: number;
  ttft_p99_ms: number;
  latency_p50_ms: number;
  latency_p99_ms: number;
  id: string;
  max_running: number;
  label: string;
};

export type BenchmarkReport = {
  recorded_at: string;
  hardware: {
    cpu: string;
    vcpu: number;
    ram_gib: number;
    device: string;
    torch: string;
    torch_threads: number;
    dtype: string;
  };
  correctness: {
    model: string;
    prompt_tokens: number;
    max_abs_logit_error: number;
    final_token_argmax_match: boolean;
    hf_attention: string;
  }[];
  throughput: {
    model: string;
    prompt_tokens: number;
    max_new_tokens: number;
    temperature: number;
    requests_per_level: number;
    block_size: number;
    num_blocks: number;
    arrival: string;
    outputs_match_across_systems: boolean;
    systems: BenchSystem[];
  };
  memory: {
    model: string;
    block_size: number;
    num_blocks: number;
    slot_budget: number;
    prompt_tokens: number;
    max_new_tokens: number;
    requests: number;
    note: string;
    systems: MemorySystem[];
  };
  speculative: {
    target: string;
    draft: string;
    gamma: number;
    prompt_tokens: number;
    max_new_tokens: number;
    temperature: number;
    greedy_match: boolean;
    generated_tokens: number;
    target_only_wall_s: number;
    speculative_wall_s: number;
    target_only_tps: number;
    speculative_tps: number;
    proposed: number;
    accepted_draft: number;
    rounds: number;
    emitted: number;
    acceptance_rate: number;
    mean_emitted_per_round: number;
  };
  elapsed_s: number;
};

export type TraceSeq = {
  id: string;
  prompt: number;
  generated: number;
  computed: number;
  blocks: number[];
  status: string;
  finish_reason: string | null;
};

export type TraceStep = {
  phases: string[];
  preempted: string[];
  waiting: string[];
  running: TraceSeq[];
  finished: string[];
  occupancy: (string | null)[];
  events: { id: string; token: number | null; phase: string; finished: boolean }[];
  free_blocks: number;
  index: number;
  arrived: string[];
  note: string;
};

export type TraceFile = {
  block_size: number;
  num_blocks: number;
  watermark_blocks: number;
  max_batch_size: number;
  max_prefill_tokens: number;
  max_new_tokens: number;
  model: {
    family: string;
    n_layer: number;
    n_embd: number;
    n_head: number;
    vocab_size: number;
    weights: string;
  };
  prompts: Record<string, number>;
  steps: TraceStep[];
};
