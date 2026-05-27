/**
 * Generic single-select card group. Built on Radix UI's RadioGroup for the
 * accessibility + keyboard nav; the visual states are driven by plain JS
 * className conditions (we don't rely on Tailwind's data-[state=...]
 * variants compiling, since that's been flaky here).
 *
 * Selected card is fully color-inverted: white background, dark text,
 * white border, inverted radio dot.
 */
import * as RadioGroup from "@radix-ui/react-radio-group";

import { cn } from "@/lib/utils";

export type RadioCardOption = {
  value: string;
  label: string;
  /** Short rightmost tag, e.g. "GPU". */
  tag?: string;
  /** Subtitle line below the label. */
  sublabel?: string;
  disabled?: boolean;
};

export function RadioCardGroup({
  value,
  onValueChange,
  options,
  disabled,
  className,
  name,
}: {
  value: string;
  onValueChange: (next: string) => void;
  options: RadioCardOption[];
  disabled?: boolean;
  className?: string;
  name?: string;
}) {
  return (
    <RadioGroup.Root
      value={value}
      onValueChange={onValueChange}
      disabled={disabled}
      name={name}
      className={cn("flex flex-col gap-2", className)}
    >
      {options.map((opt) => {
        const selected = opt.value === value;
        const isDisabled = !!opt.disabled;
        return (
          <RadioGroup.Item
            key={opt.value}
            value={opt.value}
            disabled={isDisabled}
            className={cn(
              "w-full text-left px-4 py-3 rounded-sm border-2 text-sm transition-colors flex items-center justify-between gap-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              selected
                ? "bg-card text-foreground border-foreground"
                : isDisabled
                ? "bg-muted/15 text-muted-foreground/40 border-border/30 cursor-not-allowed opacity-60"
                : "bg-card text-foreground border-border hover:border-foreground/40 hover:bg-accent cursor-pointer",
            )}
          >
            <div className="flex items-center gap-3 min-w-0">
              <span
                aria-hidden
                className={cn(
                  "relative inline-block size-4 rounded-full border-2 shrink-0",
                  selected
                    ? "border-foreground"
                    : isDisabled
                    ? "border-muted-foreground/30"
                    : "border-muted-foreground/60",
                )}
              >
                {selected && (
                  <span className="absolute inset-[3px] rounded-full bg-foreground" />
                )}
              </span>
              <div className="min-w-0">
                <div className="mono text-xs truncate">{opt.label}</div>
                {opt.sublabel && (
                  <div className="text-[10px] mt-0.5 text-muted-foreground/60">
                    {opt.sublabel}
                  </div>
                )}
              </div>
            </div>
            {opt.tag && (
              <span className="text-[10px] mono shrink-0 px-1.5 py-0.5 rounded-sm border border-border text-muted-foreground">
                {opt.tag}
              </span>
            )}
          </RadioGroup.Item>
        );
      })}
    </RadioGroup.Root>
  );
}
