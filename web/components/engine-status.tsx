"use client";

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type Status =
  | { kind: "checking" }
  | { kind: "online"; model: string }
  | { kind: "offline" };

const endpoint = process.env.NEXT_PUBLIC_NANO_INFER_URL || "http://127.0.0.1:8741";

export function EngineStatus() {
  const [status, setStatus] = useState<Status>({ kind: "checking" });
  const base = endpoint.replace(/\/$/, "");

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 2000);
    fetch(`${base}/health`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("unhealthy");
        const body = (await response.json()) as { model?: string };
        if (!cancelled) setStatus({ kind: "online", model: body.model || "unknown" });
      })
      .catch(() => {
        if (!cancelled) setStatus({ kind: "offline" });
      })
      .finally(() => window.clearTimeout(timer));
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [base]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <CardTitle>Local engine</CardTitle>
          {status.kind === "checking" ? <Badge variant="outline">checking</Badge> : null}
          {status.kind === "online" ? <Badge variant="mint">online</Badge> : null}
          {status.kind === "offline" ? <Badge variant="coral">offline</Badge> : null}
        </div>
        <CardDescription>
          Optional. Charts and the tape read the recorded JSON and do not need a running server.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {status.kind === "checking" ? (
          <p className="text-sm text-muted">Looking for an engine at {base}.</p>
        ) : null}
        {status.kind === "online" ? (
          <p className="text-sm text-foreground">
            Health check returned ok. Loaded model <span className="font-mono">{status.model}</span>.
          </p>
        ) : null}
        {status.kind === "offline" ? (
          <p className="text-sm text-coral">
            No engine responded at {base}. Start one with <span className="font-mono">python -m nano_infer.server</span>{" "}
            or set <span className="font-mono">NEXT_PUBLIC_NANO_INFER_URL</span>. The recorded benchmarks stay
            available.
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}
