/**
 * Inline rename. Click → input appears in place; Enter saves; Esc cancels;
 * blur saves. While saving, the input dims and is disabled.
 *
 * Pure presentation — the parent passes `value` + an async `onSave`.
 * Stays generic so it can rename anything (project, run, policy, …).
 */
import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";

export function InlineRename({
  value,
  onSave,
  className,
  inputClassName,
  triggerClassName,
  placeholder = "Untitled",
}: {
  value: string;
  onSave: (next: string) => Promise<void> | void;
  className?: string;
  inputClassName?: string;
  triggerClassName?: string;
  placeholder?: string;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editing) setDraft(value);
  }, [value, editing]);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const commit = async () => {
    const next = draft.trim();
    if (!next || next === value) {
      setDraft(value);
      setEditing(false);
      return;
    }
    setSaving(true);
    try {
      await onSave(next);
    } finally {
      setSaving(false);
      setEditing(false);
    }
  };

  if (!editing) {
    return (
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          setEditing(true);
        }}
        className={cn(
          "w-full text-left truncate hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm",
          triggerClassName,
          className,
        )}
        title="Click to rename"
      >
        {value || placeholder}
      </button>
    );
  }

  return (
    <input
      ref={inputRef}
      value={draft}
      disabled={saving}
      autoFocus
      onChange={(e) => setDraft(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Enter") {
          e.preventDefault();
          void commit();
        } else if (e.key === "Escape") {
          e.preventDefault();
          setDraft(value);
          setEditing(false);
        }
      }}
      onBlur={() => void commit()}
      placeholder={placeholder}
      className={cn(
        "w-full bg-transparent text-foreground border border-border rounded-sm px-2 py-0.5 focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50",
        inputClassName,
        className,
      )}
    />
  );
}
