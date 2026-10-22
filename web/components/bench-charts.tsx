"use client";

import { useEffect, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { oneDecimal } from "@/lib/format";
import { SYSTEM_COLOR } from "@/lib/palette";
import type { BenchSystem, Level } from "@/lib/types";

type Percentile = "p50" | "p99";

type TipEntry = {
  dataKey?: string | number;
  value?: number | string;
  color?: string;
};

function tipRows(payload: unknown): readonly TipEntry[] | undefined {
  if (!Array.isArray(payload)) return undefined;
  return payload as readonly TipEntry[];
}

function ChartTip({
  active,
  payload,
  label,
  unit,
  names,
}: {
  active?: boolean;
  payload?: readonly TipEntry[];
  label?: string | number;
  unit: string;
  names: Record<string, string>;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line bg-panel px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-mono text-muted">concurrency {label}</div>
      <div className="space-y-1">
        {payload.map((item) => {
          const key = String(item.dataKey ?? "");
          const numeric = typeof item.value === "number" ? item.value : Number(item.value);
          return (
            <div key={key} className="flex items-center justify-between gap-4 font-mono">
              <span style={{ color: item.color }}>{names[key] ?? key}</span>
              <span>
                {Number.isFinite(numeric) ? oneDecimal(numeric) : "—"} {unit}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function rowsFor(
  systems: BenchSystem[],
  enabled: string[],
  pick: (level: Level) => number,
) {
  const concurrencies = systems[0]?.levels.map((level) => level.concurrency) ?? [];
  return concurrencies.map((concurrency) => {
    const row: Record<string, number> = { concurrency };
    for (const system of systems) {
      if (!enabled.includes(system.id)) continue;
      const level = system.levels.find((item) => item.concurrency === concurrency);
      if (level) row[system.id] = pick(level);
    }
    return row;
  });
}

function EmptyPlot({ message }: { message: string }) {
  return (
    <div className="flex h-60 items-center justify-center rounded-lg border border-dashed border-line bg-panel-2 px-6 text-center text-sm text-muted sm:h-72">
      {message}
    </div>
  );
}

function ChartFrame({
  children,
  empty,
}: {
  children: ReactNode;
  empty: boolean;
}) {
  if (empty) return <EmptyPlot message="Select at least one system to plot." />;
  return <div className="h-60 w-full sm:h-72">{children}</div>;
}

export function BenchCharts({
  systems,
  model,
  promptTokens,
  newTokens,
  requests,
  arrival,
  outputsMatch,
}: {
  systems: BenchSystem[];
  model: string;
  promptTokens: number;
  newTokens: number;
  requests: number;
  arrival: string;
  outputsMatch: boolean;
}) {
  const [enabled, setEnabled] = useState<string[]>(() => systems.map((system) => system.id));
  const [percentile, setPercentile] = useState<Percentile>("p50");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setReady(true);
  }, []);

  const names = Object.fromEntries(systems.map((system) => [system.id, system.label]));
  const active = systems.filter((system) => enabled.includes(system.id));
  const ttftKey = percentile === "p50" ? "ttft_p50_ms" : "ttft_p99_ms";
  const latKey = percentile === "p50" ? "latency_p50_ms" : "latency_p99_ms";

  function toggle(id: string) {
    setEnabled((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Systems">
          {systems.map((system) => {
            const on = enabled.includes(system.id);
            return (
              <Button
                key={system.id}
                type="button"
                size="sm"
                variant={on ? "secondary" : "outline"}
                aria-pressed={on}
                onClick={() => toggle(system.id)}
              >
                <span
                  className="size-2 rounded-full"
                  style={{ background: on ? SYSTEM_COLOR[system.id] : "#2e3629" }}
                />
                {system.label}
              </Button>
            );
          })}
        </div>
        <div className="flex gap-2" role="group" aria-label="Percentile">
          {(["p50", "p99"] as const).map((item) => (
            <Button
              key={item}
              type="button"
              size="sm"
              variant={percentile === item ? "default" : "outline"}
              aria-pressed={percentile === item}
              onClick={() => setPercentile(item)}
            >
              {item}
            </Button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Throughput</CardTitle>
            <CardDescription>
              {model}, {requests} requests × {newTokens} new tokens, prompt {promptTokens},
              temperature 0. Tokens per second is generated tokens over wall time.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!ready ? (
              <div className="h-60 animate-pulse rounded-lg bg-panel-2 sm:h-72" />
            ) : (
              <ChartFrame empty={active.length === 0}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={rowsFor(systems, enabled, (level) => level.throughput_tps)}>
                    <CartesianGrid vertical={false} />
                    <XAxis
                      dataKey="concurrency"
                      tickFormatter={(value) => `c${value}`}
                      stroke="#9aa592"
                      fontSize={12}
                    />
                    <YAxis stroke="#9aa592" fontSize={12} width={48} />
                    <Tooltip
                      content={(props) => (
                        <ChartTip
                          active={props.active}
                          payload={tipRows(props.payload)}
                          label={props.label}
                          unit="tok/s"
                          names={names}
                        />
                      )}
                    />
                    {active.map((system) => (
                      <Bar
                        key={system.id}
                        dataKey={system.id}
                        fill={SYSTEM_COLOR[system.id] ?? "#c7cbb8"}
                        radius={[3, 3, 0, 0]}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Time to first token</CardTitle>
            <CardDescription>
              {percentile} across the wave. Queue wait inside the wave is included.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!ready ? (
              <div className="h-60 animate-pulse rounded-lg bg-panel-2 sm:h-72" />
            ) : (
              <ChartFrame empty={active.length === 0}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={rowsFor(systems, enabled, (level) => level[ttftKey])}>
                    <CartesianGrid vertical={false} />
                    <XAxis
                      dataKey="concurrency"
                      tickFormatter={(value) => `c${value}`}
                      stroke="#9aa592"
                      fontSize={12}
                    />
                    <YAxis stroke="#9aa592" fontSize={12} width={48} />
                    <Tooltip
                      content={(props) => (
                        <ChartTip
                          active={props.active}
                          payload={tipRows(props.payload)}
                          label={props.label}
                          unit="ms"
                          names={names}
                        />
                      )}
                    />
                    {active.map((system) => (
                      <Line
                        key={system.id}
                        type="monotone"
                        dataKey={system.id}
                        stroke={SYSTEM_COLOR[system.id] ?? "#c7cbb8"}
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </ChartFrame>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Request latency</CardTitle>
          <CardDescription>
            {percentile} from submit to the last token. With {requests} requests, p99 sits next to
            the slowest sample in the wave.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!ready ? (
            <div className="h-60 animate-pulse rounded-lg bg-panel-2 sm:h-72" />
          ) : (
            <ChartFrame empty={active.length === 0}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rowsFor(systems, enabled, (level) => level[latKey])}>
                  <CartesianGrid vertical={false} />
                  <XAxis
                    dataKey="concurrency"
                    tickFormatter={(value) => `c${value}`}
                    stroke="#9aa592"
                    fontSize={12}
                  />
                  <YAxis stroke="#9aa592" fontSize={12} width={48} />
                  <Tooltip
                    content={(props) => (
                      <ChartTip
                        active={props.active}
                        payload={tipRows(props.payload)}
                        label={props.label}
                        unit="ms"
                        names={names}
                      />
                    )}
                  />
                  {active.map((system) => (
                    <Line
                      key={system.id}
                      type="monotone"
                      dataKey={system.id}
                      stroke={SYSTEM_COLOR[system.id] ?? "#c7cbb8"}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </ChartFrame>
          )}

          {active.length === 0 ? (
            <p className="text-sm text-muted">No systems selected.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-left text-sm">
                <caption className="sr-only">
                  Throughput and latency by system and concurrency
                </caption>
                <thead className="font-mono text-[11px] uppercase tracking-wide text-muted">
                  <tr>
                    <th className="py-2 pr-3 font-medium">System</th>
                    <th className="py-2 pr-3 font-medium">c</th>
                    <th className="py-2 pr-3 font-medium">tok/s</th>
                    <th className="py-2 pr-3 font-medium">TTFT p50</th>
                    <th className="py-2 pr-3 font-medium">TTFT p99</th>
                    <th className="py-2 pr-3 font-medium">Lat p50</th>
                    <th className="py-2 pr-3 font-medium">Lat p99</th>
                    <th className="py-2 font-medium">Wall</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-xs">
                  {active.flatMap((system) =>
                    system.levels.map((level) => (
                      <tr key={`${system.id}-${level.concurrency}`} className="border-t border-line">
                        <td className="py-2 pr-3 text-foreground">{system.label}</td>
                        <td className="py-2 pr-3">{level.concurrency}</td>
                        <td className="py-2 pr-3">{oneDecimal(level.throughput_tps)}</td>
                        <td className="py-2 pr-3">{oneDecimal(level.ttft_p50_ms)}</td>
                        <td className="py-2 pr-3">{oneDecimal(level.ttft_p99_ms)}</td>
                        <td className="py-2 pr-3">{oneDecimal(level.latency_p50_ms)}</td>
                        <td className="py-2 pr-3">{oneDecimal(level.latency_p99_ms)}</td>
                        <td className="py-2">{level.wall_s.toFixed(2)}s</td>
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
            </div>
          )}
          <p className="text-sm leading-6 text-muted">{arrival}</p>
          <div className="flex flex-wrap gap-2">
            <Badge variant={outputsMatch ? "mint" : "coral"}>
              {outputsMatch ? "Greedy tokens match across systems" : "Greedy tokens diverged"}
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
