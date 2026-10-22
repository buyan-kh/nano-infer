import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { oneDecimal } from "@/lib/format";
import type { BenchmarkReport } from "@/lib/types";

export function Findings({ report }: { report: BenchmarkReport }) {
  const paged = report.memory.systems.find((system) => system.id === "paged");
  const reserve = report.memory.systems.find((system) => system.id === "reserve");
  const spec = report.speculative;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Memory pressure</CardTitle>
          <CardDescription>
            {report.memory.model}, {report.memory.requests} requests, prompt{" "}
            {report.memory.prompt_tokens} + {report.memory.max_new_tokens} new tokens,{" "}
            {report.memory.num_blocks} blocks × {report.memory.block_size} = {report.memory.slot_budget}{" "}
            slots.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm leading-6 text-muted">{report.memory.note}</p>
          {paged && reserve ? (
            <div className="grid gap-3 sm:grid-cols-2">
              {[paged, reserve].map((system) => (
                <div key={system.id} className="rounded-lg border border-line bg-panel-2 p-3">
                  <div className="font-mono text-[11px] uppercase tracking-wide text-muted">
                    {system.label}
                  </div>
                  <div className="mt-1 font-display text-3xl">{oneDecimal(system.throughput_tps)}</div>
                  <div className="font-mono text-xs text-muted">tok/s</div>
                  <p className="mt-2 text-sm text-muted">
                    max running {system.max_running} · TTFT p50 {oneDecimal(system.ttft_p50_ms)} ms ·
                    lat p50 {oneDecimal(system.latency_p50_ms)} ms
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted">Memory comparison is missing from the recorded file.</p>
          )}
          {paged && reserve ? (
            <p className="text-sm leading-6 text-foreground">
              Paging admitted {paged.max_running} sequences and ran at {oneDecimal(paged.throughput_tps)}{" "}
              tok/s. Up-front reservation admitted {reserve.max_running} and ran at{" "}
              {oneDecimal(reserve.throughput_tps)} tok/s. Replaying preempted prompts cost more than the
              extra batch width returned.
            </p>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Speculative decoding</CardTitle>
          <CardDescription>
            Target {spec.target}, draft {spec.draft}, gamma {spec.gamma}, prompt {spec.prompt_tokens} +{" "}
            {spec.max_new_tokens} tokens, temperature {spec.temperature}.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-line bg-panel-2 p-3">
              <div className="font-mono text-[11px] uppercase tracking-wide text-muted">Target only</div>
              <div className="mt-1 font-display text-3xl">{oneDecimal(spec.target_only_tps)}</div>
              <div className="font-mono text-xs text-muted">tok/s · {spec.target_only_wall_s.toFixed(2)}s</div>
            </div>
            <div className="rounded-lg border border-line bg-panel-2 p-3">
              <div className="font-mono text-[11px] uppercase tracking-wide text-muted">With draft</div>
              <div className="mt-1 font-display text-3xl">{oneDecimal(spec.speculative_tps)}</div>
              <div className="font-mono text-xs text-muted">tok/s · {spec.speculative_wall_s.toFixed(2)}s</div>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant={spec.greedy_match ? "mint" : "coral"}>
              {spec.greedy_match ? "Greedy output matches target" : "Greedy output diverged"}
            </Badge>
            <Badge variant="amber">
              {(spec.acceptance_rate * 100).toFixed(1)}% accepted
            </Badge>
          </div>
          <p className="text-sm leading-6 text-foreground">
            The draft proposed {spec.proposed} tokens and {spec.accepted_draft} were kept, across{" "}
            {spec.rounds} rounds ({oneDecimal(spec.mean_emitted_per_round)} tokens emitted per round,{" "}
            {spec.emitted} total). Acceptance is high, and the pair is still slower on CPU: distilgpt2 is
            six layers against GPT-2&apos;s twelve, which is not a cheap enough draft here.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
