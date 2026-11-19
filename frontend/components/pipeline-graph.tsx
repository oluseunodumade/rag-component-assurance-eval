"use client";

import { Background, Handle, Position, ReactFlow, useReactFlow, type Edge, type Node, type NodeMouseHandler, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { type StageStatus, type StageView, type SubView } from "@/lib/stages";

type StageNodeData = StageView & { selected: boolean };
type SubNodeData = SubView & { selected: boolean };
type FlowNode = Node<StageNodeData, "stage"> | Node<SubNodeData, "sub">;

const STATUS_CLASS: Record<StageStatus, string> = {
  pending: "border-zinc-200 bg-zinc-50 text-zinc-400",
  running: "border-zinc-900 bg-white text-zinc-950",
  pass: "border-emerald-200 bg-emerald-50 text-emerald-700",
  fail: "border-rose-200 bg-rose-50 text-rose-700",
  warn: "border-amber-200 bg-amber-50 text-amber-800",
};

function StatusBadge({ status }: { status: StageStatus }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase ${STATUS_CLASS[status]}`}>
      {status === "running" && <Loader2 className="h-3 w-3 animate-spin" aria-hidden />}
      {status === "pending" ? "Idle" : status === "running" ? "Running" : status}
    </span>
  );
}

function StageNode({ data }: NodeProps<Node<StageNodeData, "stage">>) {
  return (
    <div
      aria-label={`${data.index}. ${data.title}`}
      className={`w-[200px] rounded-xl border bg-white px-3 py-2.5 shadow-sm ${
        data.status === "running"
          ? "border-zinc-900 ring-2 ring-zinc-900/15"
          : data.selected
            ? "border-zinc-900 ring-2 ring-zinc-900/10"
            : "border-zinc-200"
      }`}
    >
      <Handle id="left" type="target" position={Position.Left} className="!h-2 !w-2 !border-zinc-300 !bg-white" />
      <Handle type="target" id="up" position={Position.Top} className="!h-2 !w-2 !border-zinc-300 !bg-white" />
      <Handle type="source" id="rise" position={Position.Top} className="!h-2 !w-2 !border-zinc-300 !bg-white" style={{ left: "70%" }} />
      <Handle id="right" type="source" position={Position.Right} className="!h-2 !w-2 !border-zinc-300 !bg-white" />
      <Handle type="source" id="down" position={Position.Bottom} className="!h-2 !w-2 !border-zinc-300 !bg-white" />
      <Handle type="target" id="down-in" position={Position.Bottom} className="!h-2 !w-2 !border-zinc-300 !bg-white" style={{ left: "70%" }} />
      <div className="flex items-center justify-between gap-2">
        <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-400">{data.index}</p>
        <StatusBadge status={data.status} />
      </div>
      <p className="mt-1.5 text-[13px] font-semibold leading-5 text-zinc-950">{data.title}</p>
      <p className="font-mono text-[10px] text-zinc-500">{data.agentName}</p>
      <div className="mt-2 flex items-center justify-between font-mono text-[11px] text-zinc-500">
        <span>{data.latencyMs === null ? "—" : `${data.latencyMs}ms`}</span>
        <span className="text-zinc-800">{data.score}</span>
      </div>
    </div>
  );
}

function SubNode({ data }: NodeProps<Node<SubNodeData, "sub">>) {
  return (
    <div
      aria-label={`${data.id} ${data.title}`}
      className={`w-[200px] rounded-lg border bg-white px-2.5 py-2 ${
        data.selected ? "border-zinc-900 ring-2 ring-zinc-900/10" : "border-zinc-200"
      }`}
    >
      <Handle type="target" position={Position.Top} className="!h-1.5 !w-1.5 !border-zinc-300 !bg-white" />
      <Handle type="source" id="down" position={Position.Bottom} className="!h-1.5 !w-1.5 !border-zinc-300 !bg-white" />
      <div className="flex items-start justify-between gap-2">
        <p className="font-mono text-[10px] text-zinc-400">{data.id}</p>
        <StatusBadge status={data.status} />
      </div>
      <p className="mt-1 text-[12px] font-medium leading-4 text-zinc-900">{data.title}</p>
      <p className="mt-1 truncate font-mono text-[10px] text-zinc-500" title={data.accelerator}>
        {data.accelerator}
      </p>
      <p className="mt-1 font-mono text-[10px] text-zinc-800">{data.score}</p>
    </div>
  );
}

const nodeTypes = { stage: StageNode, sub: SubNode };

function FitGraph({ signature }: { signature: string }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    const timer = window.setTimeout(() => {
      void fitView({ padding: 0.16 });
    }, 40);
    return () => window.clearTimeout(timer);
  }, [fitView, signature]);
  return null;
}

export function PipelineGraph({
  stages,
  activeId,
  expandedId,
  onSelect,
}: {
  stages: StageView[];
  activeId: string;
  expandedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const nodes: FlowNode[] = [];
  const edges: Edge[] = [];

  const workers = stages.filter((stage) => stage.id !== "agent");
  const supervisor = stages.find((stage) => stage.id === "agent");
  const pitch = 232;
  const rowY = 168;
  if (supervisor) {
    nodes.push({
      id: supervisor.id,
      type: "stage",
      position: { x: ((workers.length - 1) * pitch) / 2, y: 0 },
      data: { ...supervisor, selected: supervisor.id === activeId || supervisor.id === expandedId },
      draggable: false,
      style: { width: 200, background: "transparent", border: "none", padding: 0 },
    });
  }
  workers.forEach((stage, index) => {
    nodes.push({
      id: stage.id,
      type: "stage",
      position: { x: index * pitch, y: rowY },
      data: { ...stage, selected: stage.id === activeId || stage.id === expandedId },
      draggable: false,
      style: { width: 200, background: "transparent", border: "none", padding: 0 },
    });
    if (supervisor && stage.status !== "pending") {
      const retry = Boolean(stage.loopTarget);
      edges.push({
        id: `dispatch-${stage.id}`,
        source: supervisor.id,
        sourceHandle: "down",
        target: stage.id,
        targetHandle: "up",
        animated: stage.status === "running",
        label: retry ? "2 dispatches" : undefined,
        labelStyle: retry ? { fill: "#b45309", fontSize: 11 } : undefined,
        style: { stroke: retry ? "#d97706" : "#a1a1aa" },
      });
    }
    if (stage.id !== expandedId) return;
    stage.subs.forEach((sub, subIndex) => {
      nodes.push({
        id: sub.id,
        type: "sub",
        position: { x: index * pitch, y: rowY + 118 + subIndex * 96 },
        data: { ...sub, selected: sub.id === activeId },
        draggable: false,
        style: { width: 200, background: "transparent", border: "none", padding: 0 },
      });
      edges.push({
        id: `${stage.id}-${sub.id}`,
        source: subIndex === 0 ? stage.id : stage.subs[subIndex - 1].id,
        sourceHandle: "down",
        target: sub.id,
        style: { stroke: sub.status === "pending" ? "#e4e4e7" : "#d4d4d8" },
      });
    });
  });

  workers.slice(0, -1).forEach((stage, index) => {
    const next = workers[index + 1];
    edges.push({
      id: `${stage.id}-${next.id}`,
      source: stage.id,
      sourceHandle: "right",
      target: next.id,
      targetHandle: "left",
      type: "straight",
      animated: next.status === "running",
      style: { stroke: stage.status === "pending" ? "#e4e4e7" : "#d4d4d8" },
    });
  });

  const handleClick: NodeMouseHandler<FlowNode> = (_event, node) => {
    const status = node.data.status;
    if (status !== "pending") onSelect(node.id);
  };

  const openCount = stages.find((stage) => stage.id === expandedId)?.subs.length ?? 0;
  const height = rowY + (openCount === 0 ? 150 : 130 + openCount * 96);

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50" style={{ height }}>
      {mounted && <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={handleClick}
        fitView
        fitViewOptions={{ padding: 0.16 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        preventScrolling={false}
        proOptions={{ hideAttribution: false }}
        minZoom={0.25}
        maxZoom={1.2}
      >
        <FitGraph signature={expandedId ?? "none"} />
        <Background color="#d4d4d8" gap={18} size={1} />
      </ReactFlow>}
    </div>
  );
}
