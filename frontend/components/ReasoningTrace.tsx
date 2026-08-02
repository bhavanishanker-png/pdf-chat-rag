"use client";

import { useState } from "react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight, GitBranch } from "lucide-react";

export function ReasoningTrace({ steps }: { steps: string[] }) {
  const [open, setOpen] = useState(false);

  if (!steps.length) return null;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors mt-2">
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
        <GitBranch className="h-3.5 w-3.5" />
        Reasoning trace ({steps.length} steps)
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2">
        <ol className="flex flex-col gap-1 border-l-2 border-border pl-3">
          {steps.map((step, i) => (
            <li key={i} className="flex items-start gap-1.5 text-xs text-muted-foreground">
              <span className="shrink-0 font-mono text-primary/70">{i + 1}.</span>
              {step}
            </li>
          ))}
        </ol>
      </CollapsibleContent>
    </Collapsible>
  );
}
