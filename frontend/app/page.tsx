"use client";

import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { ChatInterface } from "@/components/ChatInterface";

export default function Home() {
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [topK, setTopK] = useState(5);
  const [clearKey, setClearKey] = useState(0);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar
        chunkSize={chunkSize}
        chunkOverlap={chunkOverlap}
        topK={topK}
        onChunkSize={setChunkSize}
        onChunkOverlap={setChunkOverlap}
        onTopK={setTopK}
        onDBCleared={() => setClearKey((k) => k + 1)}
      />
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <ChatInterface key={clearKey} topK={topK} />
      </main>
    </div>
  );
}
