"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { AlertTriangle, BarChart3, CloudUpload, Crosshair, Database, FileText, Plus, ShieldCheck, Upload, Workflow } from "lucide-react";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import { getCatalogAttacks, getCatalogDatasets, getModelVersions, listRuns } from "@/lib/api";
import { buildRunDecisionView, formatRatio } from "@/lib/reportMetrics";

const PIPELINE_STEPS = [
  { step: 1, title: "Cấu hình bài toán", desc: "Nạp mô hình & dữ liệu gốc", icon: Database, route: "/experiments/new" },
  { step: 2, title: "Cấu hình tấn công", desc: "Sinh biến thể đối kháng & OOD", icon: Crosshair, route: "/experiments/EXP-2025-0512-001/attack" },
  { step: 3, title: "Đánh giá Robustness", desc: "Phân tích suy giảm hiệu năng", icon: BarChart3, route: "/benchmark" },
  { step: 4, title: "Tôi luyện phòng thủ", desc: "Fine-tune & vá lỗi nhận diện", icon: ShieldCheck, route: "/defense" },
  { step: 5, title: "Báo cáo & Chứng nhận", desc: "Xuất báo cáo khoa học & Audit", icon: FileText, route: "/analysis" },
];

export default function DashboardView() {
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [attacks, setAttacks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    Promise.allSettled([listRuns(), getModelVersions(), getCatalogDatasets(), getCatalogAttacks()]).then((results) => {
      if (!mounted) return;
      const [runsRes, modelsRes, datasetsRes, attacksRes] = results;
      if (runsRes.status === "fulfilled" && Array.isArray(runsRes.value)) setRuns(runsRes.value);
      if (modelsRes.status === "fulfilled" && Array.isArray(modelsRes.value)) setModels(modelsRes.value);
      if (datasetsRes.status === "fulfilled" && Array.isArray(datasetsRes.value)) setDatasets(datasetsRes.value);
      if (attacksRes.status === "fulfilled" && Array.isArray(attacksRes.value)) setAttacks(attacksRes.value);
      setIsLoading(false);
    });
    return () => { mounted = false; };
  }, []);

  const completedRuns = runs.filter((run) => run.status === "COMPLETED" && run.report);
  const latestMeasuredRun = [...completedRuns].reverse().find((run) => buildRunDecisionView(run.report).dataState === "MEASURED");
  const latestDecision = latestMeasuredRun ? buildRunDecisionView(latestMeasuredRun.report) : null;
  const highRiskRun = runs.some((run) => run.report && (run.report.asr || run.report.metrics?.robustness?.attack_success_rate || 0) > 0.4);

  return (
    <div className="space-y-5 animate-fade-in">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-4">
        <Card className="xl:col-span-3" title="Quy trình đánh giá & phòng thủ AI đối kháng" subtitle="Từ cấu hình bài toán đến báo cáo và huấn luyện phòng thủ">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
            {PIPELINE_STEPS.map(({ step, title, desc, icon: Icon, route }) => (
              <Link key={step} href={route} className="group rounded-lg border border-slate-200 bg-slate-50/70 p-2.5 transition-all hover:border-blue-300 hover:bg-blue-50">
                <div className="mb-2 flex items-center justify-between"><span className="flex h-7 w-7 items-center justify-center rounded-md border border-slate-200 bg-white text-blue-600 group-hover:bg-blue-600 group-hover:text-white"><Icon className="h-3.5 w-3.5" /></span><span className="text-[10px] font-bold text-slate-400">0{step}</span></div>
                <div className="text-[11px] font-bold leading-tight text-slate-800 group-hover:text-blue-700">{title}</div><p className="mt-1.5 line-clamp-2 text-[10px] leading-snug text-slate-500">{desc}</p>
              </Link>
            ))}
          </div>
        </Card>

        <Card title="Thao tác nhanh" subtitle="Lối tắt hành động thường dùng">
          <div className="grid grid-cols-2 gap-2">
            <Link href="/experiments/new" className="rounded-lg border border-blue-200 bg-blue-50 p-2.5 text-center text-xs font-semibold text-blue-700 hover:bg-blue-100"><Plus className="mx-auto mb-1 h-4 w-4" />Tạo thí nghiệm</Link>
            <Link href="/experiments/new" className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-center text-xs font-semibold text-slate-700 hover:bg-slate-100"><Upload className="mx-auto mb-1 h-4 w-4" />Nạp mô hình</Link>
            <Link href="/experiments/new" className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-center text-xs font-semibold text-slate-700 hover:bg-slate-100"><CloudUpload className="mx-auto mb-1 h-4 w-4" />Nạp dataset</Link>
            <Link href="/analysis" className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-center text-xs font-semibold text-slate-700 hover:bg-slate-100"><FileText className="mx-auto mb-1 h-4 w-4" />Xuất báo cáo</Link>
          </div>
          <Link href="/defense"><Button variant="outline" size="sm" className="mt-3 w-full justify-center text-xs"><Workflow className="mr-1.5 h-3.5 w-3.5 text-blue-600" />Quy trình phòng thủ đóng loop</Button></Link>
        </Card>
      </div>

      <Card title="Kết quả đã xác minh" subtitle="Chỉ hiển thị số liệu có report đo được; dữ liệu chưa xác minh được giữ là — / No verified data">
        {!latestDecision ? <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm text-slate-500">{isLoading ? "Đang tải dữ liệu..." : "— / No verified data"}</div> : <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3"><p className="text-xs text-slate-500">Run gần nhất</p><p className="mt-1 font-mono text-sm font-semibold text-blue-700">{latestMeasuredRun.id}</p></div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3"><p className="text-xs text-slate-500">{latestDecision.primary.label}</p><p className="mt-1 text-xl font-bold text-slate-900">{formatRatio(latestDecision.primary.clean)}</p></div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3"><p className="text-xs text-slate-500">Hiệu năng giữ lại</p><p className="mt-1 text-xl font-bold text-emerald-600">{latestDecision.robustnessRetained.toFixed(1)}%</p></div>
        </div>}
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Mô hình đang khả dụng" subtitle="Checkpoint catalog đã đăng ký" headerAction={<Link href="/experiments/new" className="text-xs font-semibold text-blue-600">Cấu hình test →</Link>}>
          {models.length === 0 ? <p className="py-5 text-center text-xs text-slate-500">{isLoading ? "Đang tải mô hình..." : "— / No verified data"}</p> : <div className="grid gap-2 sm:grid-cols-2">{models.map((model) => <div key={model.id} className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50/70 px-3 py-2"><div className="min-w-0"><p className="truncate text-xs font-semibold text-slate-800">{model.id}</p><p className="text-[10px] text-slate-500">{model.model_name} · {model.task}</p></div><Badge variant={model.runnable ? "success" : "secondary"}>{model.runnable ? "Sẵn sàng" : "Chưa sẵn sàng"}</Badge></div>)}</div>}
        </Card>
        <Card title="Dataset đang khả dụng" subtitle="Các tập dữ liệu đã qua gate anonymization" headerAction={<Link href="/experiments/new" className="text-xs font-semibold text-blue-600">Chọn dataset →</Link>}>
          {datasets.length === 0 ? <p className="py-5 text-center text-xs text-slate-500">{isLoading ? "Đang tải dataset..." : "— / No verified data"}</p> : <div className="grid gap-2 sm:grid-cols-2">{datasets.map((dataset) => <div key={dataset.name} className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50/70 px-3 py-2"><div className="min-w-0"><p className="truncate text-xs font-semibold text-slate-800">{dataset.title || dataset.name}</p><p className="text-[10px] text-slate-500">{dataset.task_id} · {dataset.modality}</p></div><Badge variant={dataset.anonymized ? "success" : "secondary"}>{dataset.anonymized ? "Anonymized" : "Pending"}</Badge></div>)}</div>}
        </Card>
      </div>

      <Card title="Cảnh báo & Khuyến nghị" subtitle="Tín hiệu chỉ dựa trên các phiên đã ghi nhận">
        {highRiskRun ? <div className="flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50/60 p-3"><AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-500" /><div><p className="text-xs font-semibold text-slate-800">Suy giảm hiệu năng nghiêm trọng</p><p className="mt-1 text-[11px] leading-snug text-slate-600">Phát hiện phiên thử nghiệm có ASR vượt ngưỡng 40%. Đề xuất đưa vào Retraining Backlog.</p></div></div> : <div className="py-6 text-center text-xs font-medium text-slate-400">Chưa có cảnh báo nào</div>}
      </Card>

      <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-4 py-3 text-xs text-slate-500"><span>{attacks.length ? `${attacks.length} attack methods trong catalog` : "Attack catalog chưa có dữ liệu"}</span><Link href="/benchmark" className="font-semibold text-blue-600 hover:text-blue-800">Mở benchmark →</Link></div>
    </div>
  );
}
