import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2 py-0.5 font-mono text-[11px] uppercase tracking-wide",
  {
    variants: {
      variant: {
        outline: "border-line text-muted",
        amber: "border-amber/40 bg-amber/15 text-amber",
        mint: "border-mint/40 bg-mint/15 text-mint",
        coral: "border-coral/40 bg-coral/15 text-coral",
        sky: "border-sky/40 bg-sky/15 text-sky",
        plum: "border-plum/40 bg-plum/15 text-plum",
      },
    },
    defaultVariants: {
      variant: "outline",
    },
  },
);

function Badge({
  className,
  variant,
  ...props
}: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant, className }))} {...props} />;
}

export { Badge, badgeVariants };
