import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { FileText } from "lucide-react";
import type { Source } from "@/lib/api";

export function SourceCard({ source, index }: { source: Source; index: number }) {
  return (
    <Card className="bg-muted/40 text-sm">
      <CardHeader className="flex flex-row items-center gap-2 pb-2 pt-3 px-4">
        <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
        <span className="truncate font-medium text-xs">{source.source}</span>
        <div className="ml-auto flex gap-1.5">
          <Badge variant="outline" className="text-xs">p.{source.page}</Badge>
          <Badge variant="secondary" className="text-xs">#{index + 1}</Badge>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-3">
        <p className="text-xs leading-relaxed text-muted-foreground line-clamp-4">
          {source.text}
        </p>
      </CardContent>
    </Card>
  );
}
