import { BenchCharts } from "@/components/bench-charts";
import { EngineStatus } from "@/components/engine-status";
import { Findings } from "@/components/findings";
import { SchedulerPlayer } from "@/components/scheduler-player";
import { Badge } from "@/components/ui/badge";
import { cpuLabel, levelAt, oneDecimal, ratio, recordedLabel } from "@/lib/format";
import type { BenchmarkReport, TraceFile } from "@/lib/types";
import benchmarksJson from "@/data/benchmarks.json";
import traceJson from "@/data/trace.json";

const report = benchmarksJson as BenchmarkReport;
const trace = traceJson as TraceFile;

export default function Home() {
  const hardware = report.hardware;
  const systems = report.throughput.systems;
  const batched = systems.find((system) => system.id === "nano-infer");
  const serial = systems.find((system) => system.id === "serial_kv");
  const recompute = systems.find((system) => system.id === "recompute");
  const batched8 = batched ? levelAt(batched.levels, 8) : undefined;
  const serial8 = serial ? levelAt(serial.levels, 8) : undefined;
  const recompute8 = recompute ? levelAt(recompute.levels, 8) : undefined;
  const maxError = Math.max(...report.correctness.map((row) => row.max_abs_logit_error));

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-8 sm:px-6 sm:py-10">
      <header className="flex flex-col gap-6 border-b border-line pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-amber">
            Inference console
          </p>
          <h1 className="font-display text-5xl tracking-tight text-foreground sm:text-6xl">nano-infer</h1>
          <p className="mt-3 max-w-xl text-base leading-7 text-muted">
            A decoder-only server with a paged KV cache, a continuous-batching scheduler, and
            speculative decoding. Every figure on this page is a wall-clock measurement.
          </p>
        </div>
        <div className="font-mono text-xs leading-5 text-muted sm:text-right">
          <div>{cpuLabel(hardware.cpu)}</div>
          <div>
            {hardware.vcpu} vCPU · {hardware.ram_gib} GiB · {hardware.dtype}
          </div>
          <div>
            torch {hardware.torch} · {hardware.torch_threads} threads
          </div>
          <div>{recordedLabel(report.recorded_at)}</div>
        </div>
      </header>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label="Headline measurements">
        <article className="rounded-xl border border-line bg-panel px-4 py-3">
          <h2 className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">Throughput</h2>
          <p className="mt-1 font-display text-4xl">
            {batched8 ? oneDecimal(batched8.throughput_tps) : "—"}
          </p>
          <p className="mt-1 text-sm text-muted">
            tok/s at concurrency 8
            {batched8 && recompute8 ? ` · ${ratio(batched8.throughput_tps, recompute8.throughput_tps)} recompute` : ""}
          </p>
        </article>
        <article className="rounded-xl border border-line bg-panel px-4 py-3">
          <h2 className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">Versus serial KV</h2>
          <p className="mt-1 font-display text-4xl">
            {batched8 && serial8 ? ratio(batched8.throughput_tps, serial8.throughput_tps) : "—"}
          </p>
          <p className="mt-1 text-sm text-muted">
            {serial8 ? `${oneDecimal(serial8.throughput_tps)} tok/s serial KV at c8` : "missing series"}
          </p>
        </article>
        <article className="rounded-xl border border-line bg-panel px-4 py-3">
          <h2 className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">TTFT p50</h2>
          <p className="mt-1 font-display text-4xl">
            {batched8 ? oneDecimal(batched8.ttft_p50_ms) : "—"}
          </p>
          <p className="mt-1 text-sm text-muted">
            ms at concurrency 8
            {recompute8 ? ` · recompute ${oneDecimal(recompute8.ttft_p50_ms)} ms` : ""}
          </p>
        </article>
        <article className="rounded-xl border border-line bg-panel px-4 py-3">
          <h2 className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">Logit error</h2>
          <p className="mt-1 font-display text-4xl">{maxError === 0 ? "0.0" : oneDecimal(maxError)}</p>
          <p className="mt-1 text-sm text-muted">max abs vs eager attention, 16-token prompts</p>
        </article>
      </section>

      <section className="flex flex-wrap gap-2" aria-label="Correctness">
        {report.correctness.map((row) => (
          <Badge key={row.model} variant={row.final_token_argmax_match ? "mint" : "coral"}>
            {row.model} · err {row.max_abs_logit_error} · {row.hf_attention}
          </Badge>
        ))}
      </section>

      <BenchCharts
        systems={systems}
        model={report.throughput.model}
        promptTokens={report.throughput.prompt_tokens}
        newTokens={report.throughput.max_new_tokens}
        requests={report.throughput.requests_per_level}
        arrival={report.throughput.arrival}
        outputsMatch={report.throughput.outputs_match_across_systems}
      />

      <Findings report={report} />
      <SchedulerPlayer trace={trace} />
      <EngineStatus />
    </main>
  );
}
