"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { SourceCard } from "./SourceCard";
import { ReasoningTrace } from "./ReasoningTrace";
import { ask, type AskResponse } from "@/lib/api";
import { Send, Download, Bot, User, Loader2 } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: AskResponse["sources"];
  steps?: string[];
}

export function ChatInterface({ topK }: { topK: number }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const result = await ask(query, topK);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources,
          steps: result.steps,
        },
      ]);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Request failed");
      setMessages((prev) => prev.slice(0, -1));
      setInput(query);
    } finally {
      setLoading(false);
    }
  };

  const downloadChat = () => {
    const text = messages
      .map((m) => `[${m.role.toUpperCase()}]\n${m.content}`)
      .join("\n\n---\n\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "chat.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-border px-6 py-3">
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium">Chat</span>
          {messages.length > 0 && (
            <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
              {messages.filter((m) => m.role === "user").length} questions
            </span>
          )}
        </div>
        {messages.length > 0 && (
          <Button variant="ghost" size="sm" onClick={downloadChat} className="gap-1.5 text-xs">
            <Download className="h-3.5 w-3.5" />
            Download
          </Button>
        )}
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 px-6">
        <div className="flex flex-col gap-6 py-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center gap-3 py-20 text-center">
              <Bot className="h-10 w-10 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground">
                Upload a PDF in the sidebar, then ask a question.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
              <div
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
                  msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                }`}
              >
                {msg.role === "user" ? (
                  <User className="h-4 w-4" />
                ) : (
                  <Bot className="h-4 w-4 text-muted-foreground" />
                )}
              </div>

              <div className={`flex max-w-[80%] flex-col gap-3 ${msg.role === "user" ? "items-end" : ""}`}>
                <div
                  className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "rounded-tr-sm bg-primary text-primary-foreground"
                      : "rounded-tl-sm bg-muted"
                  }`}
                >
                  {msg.content}
                </div>

                {msg.role === "assistant" && (
                  <>
                    {msg.steps && <ReasoningTrace steps={msg.steps} />}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="flex flex-col gap-2 w-full">
                        <Separator />
                        <p className="text-xs font-medium text-muted-foreground">Sources</p>
                        {msg.sources.map((src, j) => (
                          <SourceCard key={j} source={src} index={j} />
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="flex gap-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted">
                <Bot className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm bg-muted px-4 py-3">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Thinking…</span>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-border px-6 py-4">
        <div className="flex gap-2">
          <Textarea
            placeholder="Ask a question about your PDF…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            rows={2}
            className="resize-none text-sm"
            disabled={loading}
          />
          <Button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            size="icon"
            className="h-auto self-stretch"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
        <p className="mt-1.5 text-xs text-muted-foreground">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  );
}
