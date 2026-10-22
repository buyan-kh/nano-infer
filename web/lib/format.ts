export function oneDecimal(value: number): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

export function ratio(numerator: number, denominator: number): string {
  if (denominator === 0) return "—";
  return `${(numerator / denominator).toFixed(2)}×`;
}

export function cpuLabel(cpu: string): string {
  return cpu.replaceAll("(R)", "").replaceAll("(TM)", "").replace(/\s+/g, " ").trim();
}

export function recordedLabel(iso: string): string {
  const formatted = new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(new Date(iso));
  return `${formatted} UTC`;
}

export function levelAt<T extends { concurrency: number }>(levels: T[], concurrency: number): T {
  const found = levels.find((level) => level.concurrency === concurrency);
  if (!found) {
    throw new Error(`missing concurrency ${concurrency}`);
  }
  return found;
}
