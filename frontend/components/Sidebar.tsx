"use client";

import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { clearDB, ingestPDF } from "@/lib/api";
import { CloudUpload, Trash2, FileText, Loader2 } from "lucide-react";

interface Props {
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
  onChunkSize: (v: number) => void;
  onChunkOverlap: (v: number) => void;
  onTopK: (v: number) => void;
  onDBCleared: () => void;
}

export function Sidebar({
  chunkSize,
  chunkOverlap,
  topK,
  onChunkSize,
  onChunkOverlap,
  onTopK,
  onDBCleared,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [ingestedFiles, setIngestedFiles] = useState<string[]>([]);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith(".pdf")) {
        toast.error("Only PDF files are supported");
        return;
      }
      setIngesting(true);
      try {
        const result = await ingestPDF(file, chunkSize, chunkOverlap);
        setIngestedFiles((prev) => [...prev, file.name]);
        toast.success(result.message ?? "PDF ingested successfully");
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Ingest failed");
      } finally {
        setIngesting(false);
      }
    },
    [chunkSize, chunkOverlap],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleClear = async () => {
    setClearing(true);
    try {
      await clearDB();
      setIngestedFiles([]);
      onDBCleared();
      toast.success("Vector store cleared");
    } catch {
      toast.error("Failed to clear DB");
    } finally {
      setClearing(false);
    }
  };

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col gap-6 border-r border-border bg-card px-5 py-6">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">PDF RAG Chat</h1>
        <p className="text-xs text-muted-foreground">
          Corrective RAG · LangGraph
        </p>
      </div>

      <Separator />

      {/* Upload */}
      <div className="flex flex-col gap-3">
        <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
          Upload
        </p>
        <div
          role="button"
          tabIndex={0}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors ${
            dragging
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-muted/40"
          }`}
        >
          {ingesting ? (
            <Loader2 className="h-7 w-7 animate-spin text-primary" />
          ) : (
            <CloudUpload className="h-7 w-7 text-muted-foreground" />
          )}
          <span className="text-xs text-muted-foreground">
            {ingesting ? "Ingesting…" : "Drop a PDF or click to browse"}
          </span>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = "";
          }}
        />

        {ingestedFiles.length > 0 && (
          <div className="flex flex-col gap-1">
            {ingestedFiles.map((name) => (
              <div
                key={name}
                className="flex items-center gap-2 rounded-md bg-muted px-2 py-1"
              >
                <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <span className="truncate text-xs">{name}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <Separator />

      {/* Settings */}
      <div className="flex flex-col gap-5">
        <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
          Settings
        </p>

        <SliderField
          label="Chunk size"
          value={chunkSize}
          min={200}
          max={4000}
          step={100}
          onChange={onChunkSize}
        />
        <SliderField
          label="Chunk overlap"
          value={chunkOverlap}
          min={0}
          max={800}
          step={50}
          onChange={onChunkOverlap}
        />
        <SliderField
          label="Top-K results"
          value={topK}
          min={1}
          max={10}
          step={1}
          onChange={onTopK}
        />
      </div>

      <Separator />

      {/* Actions */}
      <div className="flex flex-col gap-2">
        <Button
          variant="destructive"
          size="sm"
          onClick={handleClear}
          disabled={clearing}
          className="gap-2"
        >
          {clearing ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Trash2 className="h-4 w-4" />
          )}
          Clear Vector DB
        </Button>
      </div>

      <div className="mt-auto flex flex-wrap gap-1">
        <Badge variant="secondary">LangGraph</Badge>
        <Badge variant="secondary">ChromaDB</Badge>
        <Badge variant="secondary">Groq</Badge>
        <Badge variant="secondary">BM25</Badge>
      </div>
    </aside>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <label className="text-xs text-muted-foreground">{label}</label>
        <span className="rounded-md bg-muted px-1.5 py-0.5 text-xs font-mono">
          {value}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : (v as number))}
        className="w-full"
      />
    </div>
  );
}
