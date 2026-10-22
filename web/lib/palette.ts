export const SYSTEM_COLOR: Record<string, string> = {
  "nano-infer": "#e6a15c",
  serial_kv: "#7eb6d9",
  recompute: "#e07a6a",
  paged: "#8fd6b4",
  reserve: "#e6a15c",
};

export const SEQ_COLOR: Record<string, string> = {
  A: "#e6a15c",
  B: "#7eb6d9",
  C: "#8fd6b4",
  D: "#d58bd0",
};

export function sequenceColor(id: string): string {
  return SEQ_COLOR[id] ?? "#c7cbb8";
}
