"use client";

import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { sequenceColor } from "@/lib/palette";
import type { TraceFile, TraceStep } from "@/lib/types";

const SPEEDS = [1, 2, 4] as const;

function phaseVariant(phase: string): "amber" | "mint" | "coral" | "plum" | "outline" {
  if (phase === "prefill") return "amber";
  if (phase === "decode") return "mint";
  if (phase === "preempt") return "coral";
  if (phase === "speculative") return "plum";
  return "outline";
}

function filledSlots(step: TraceStep, blockSize: number) {
  const owned = new Map<number, { id: string; slots: boolean[] }>();
  for (const seq of step.running) {
    seq.blocks.forEach((physical, logical) => {
      const start = logical * blockSize;
      const slots = Array.from({ length: blockSize }, (_, slot) => start + slot < seq.computed);
      owned.set(physical, { id: seq.id, slots });
    });
  }
  return owned;
}

export function SchedulerPlayer({ trace }: { trace: TraceFile }) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const [ready, setReady] = useState(false);
  const last = Math.max(trace.steps.length - 1, 0);
  const step = trace.steps[index];

  useEffect(() => {
    setReady(true);
  }, []);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setIndex((current) => (current >= last ? current : current + 1));
    }, 800 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, last]);

  useEffect(() => {
    if (playing && index >= last) setPlaying(false);
  }, [playing, index, last]);

  if (!ready || !step) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Scheduler tape</CardTitle>
          <CardDescription>Loading the recorded allocation trace.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 animate-pulse rounded-lg bg-panel-2" />
        </CardContent>
      </Card>
    );
  }

  const owned = filledSlots(step, trace.block_size);
  const promptLine = Object.entries(trace.prompts)
    .map(([id, length]) => `${id} ${length}`)
    .join(" · ");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Scheduler tape</CardTitle>
        <CardDescription>
          Playback of a real scheduler run on a tiny random GPT-2 ({trace.model.n_layer} layers,{" "}
          {trace.model.n_embd} wide, vocab {trace.model.vocab_size}). {trace.num_blocks} blocks ×{" "}
          {trace.block_size} slots. Prompts {promptLine}. Watermark {trace.watermark_blocks}, batch
          cap {trace.max_batch_size}, prefill budget {trace.max_prefill_tokens}, {trace.max_new_tokens}{" "}
          new tokens.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="icon"
              variant="outline"
              aria-label="Previous step"
              disabled={index === 0}
              onClick={() => {
                setPlaying(false);
                setIndex((current) => Math.max(0, current - 1));
              }}
            >
              <ChevronLeft />
            </Button>
            <Button
              type="button"
              size="icon"
              aria-label={playing ? "Pause" : "Play"}
              onClick={() => {
                if (index >= last) setIndex(0);
                setPlaying((current) => !current);
              }}
            >
              {playing ? <Pause /> : <Play />}
            </Button>
            <Button
              type="button"
              size="icon"
              variant="outline"
              aria-label="Next step"
              disabled={index >= last}
              onClick={() => {
                setPlaying(false);
                setIndex((current) => Math.min(last, current + 1));
              }}
            >
              <ChevronRight />
            </Button>
          </div>
          <Slider
            className="flex-1"
            aria-label="Scheduler step"
            min={0}
            max={last}
            step={1}
            value={[index]}
            onValueChange={(value) => {
              const next = value[0];
              if (next === undefined) return;
              setPlaying(false);
              setIndex(next);
            }}
          />
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-muted">
              {index + 1}/{trace.steps.length}
            </span>
            <div className="flex gap-1" role="group" aria-label="Playback speed">
              {SPEEDS.map((item) => (
                <Button
                  key={item}
                  type="button"
                  size="sm"
                  variant={speed === item ? "secondary" : "ghost"}
                  aria-pressed={speed === item}
                  onClick={() => setSpeed(item)}
                >
                  {item}×
                </Button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {step.phases.map((phase) => (
            <Badge key={phase} variant={phaseVariant(phase)}>
              {phase}
            </Badge>
          ))}
          <span className="font-mono text-xs text-muted">{step.free_blocks} free blocks</span>
        </div>
        <p className="font-mono text-sm text-foreground">{step.note}</p>

        {step.preempted.length > 0 ? (
          <p className="rounded-md border border-coral/40 bg-coral/10 px-3 py-2 text-sm text-coral">
            {step.preempted.join(", ")} dropped KV this step and will replay from the prompt. Generated
            tokens are kept.
          </p>
        ) : null}

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_240px]">
          <div>
            <h3 className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
              KV blocks
            </h3>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-8">
              {Array.from({ length: trace.num_blocks }, (_, physical) => {
                const info = owned.get(physical);
                const color = info ? sequenceColor(info.id) : undefined;
                return (
                  <div
                    key={physical}
                    className="rounded-md border bg-panel-2 p-2"
                    style={{ borderColor: color ?? "#2e3629" }}
                  >
                    <div className="mb-1 flex items-center justify-between font-mono text-[10px] text-muted">
                      <span>{physical}</span>
                      <span style={{ color }}>{info?.id ?? "free"}</span>
                    </div>
                    <div
                      className="grid gap-1"
                      style={{ gridTemplateColumns: `repeat(${trace.block_size}, minmax(0, 1fr))` }}
                    >
                      {Array.from({ length: trace.block_size }, (_, slot) => {
                        const on = info?.slots[slot] ?? false;
                        return (
                          <div
                            key={slot}
                            className="aspect-square rounded-[2px] motion-safe:transition-colors motion-safe:duration-300"
                            style={{
                              background: on && color ? color : "#10140f",
                              boxShadow: on ? undefined : "inset 0 0 0 1px #2e3629",
                            }}
                            title={info ? `${info.id} slot ${slot}${on ? " filled" : " reserved"}` : "free"}
                          />
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <h3 className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
                Queue
              </h3>
              {step.waiting.length === 0 ? (
                <p className="text-sm text-muted">Queue empty.</p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {step.waiting.map((id) => (
                    <Badge key={id} variant={step.preempted.includes(id) ? "coral" : "sky"}>
                      <span
                        className="mr-1 inline-block size-1.5 rounded-full"
                        style={{ background: sequenceColor(id) }}
                      />
                      {id}
                      {step.preempted.includes(id) ? " preempted" : " waiting"}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
            <div>
              <h3 className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
                Running
              </h3>
              {step.running.length === 0 ? (
                <p className="text-sm text-muted">Nothing running.</p>
              ) : (
                <ul className="space-y-2">
                  {step.running.map((seq) => (
                    <li key={seq.id} className="rounded-md border border-line px-2 py-1.5">
                      <div className="flex items-center gap-2 font-mono text-sm">
                        <span
                          className="size-2 rounded-full"
                          style={{ background: sequenceColor(seq.id) }}
                        />
                        {seq.id}
                        <span className="text-muted">
                          {seq.generated}/{trace.max_new_tokens}
                        </span>
                      </div>
                      <p className="mt-1 font-mono text-[11px] text-muted">
                        kv {seq.computed} · blocks {seq.blocks.join(",")}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div>
              <h3 className="mb-2 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
                Finished
              </h3>
              {step.finished.length === 0 ? (
                <p className="text-sm text-muted">None finished yet.</p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {step.finished.map((id) => (
                    <Badge key={id} variant="mint">
                      {id}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {step.events.length === 0 ? (
            <span className="text-sm text-muted">No tokens this step.</span>
          ) : (
            step.events.map((event, eventIndex) => (
              <Badge key={`${event.id}-${eventIndex}`} variant={phaseVariant(event.phase)}>
                {event.id} · {event.phase}
                {event.finished ? " · done" : ""}
              </Badge>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
