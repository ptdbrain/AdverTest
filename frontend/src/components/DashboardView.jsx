"use client";

import {
  AlertTriangle,
  CloudUpload,
  Database,
  FileText,
  Plus,
  Upload,
  Workflow,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import Card from "@/components/common/Card";
import QuickUploadDialog from "@/components/dashboard/QuickUploadDialog";
import { useProject } from "@/context/ProjectContext";
import { getCatalogAttacks, getCatalogDatasets, getModelVersions, listRuns } from "@/lib/api";

export default function DashboardView() {
  const { activeProject, activeProjectId } = useProject();
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [attacks, setAttacks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [uploadKind, setUploadKind] = useState(null); // null | "model" | "dataset"
  const mountedRef = useRef(true);
  useEffect(
    () => () => {
      mountedRef.current = false;
    },
    [],
  );

  const loadAssets = useCallback(async () => {
    const results = await Promise.allSettled([
      listRuns(activeProjectId || undefined),
      getModelVersions(),
      getCatalogDatasets(),
      getCatalogAttacks(),
    ]);
    if (!mountedRef.current) return;
    const [runsRes, modelsRes, datasetsRes, attacksRes] = results;
    if (runsRes.status === "fulfilled" && Array.isArray(runsRes.value)) setRuns(runsRes.value);
    if (modelsRes.status === "fulfilled" && Array.isArray(modelsRes.value)) setModels(modelsRes.value);
    if (datasetsRes.status === "fulfilled" && Array.isArray(datasetsRes.value)) setDatasets(datasetsRes.value);
    if (attacksRes.status === "fulfilled" && Array.isArray(attacksRes.value)) setAttacks(attacksRes.value);
    setIsLoading(false);
  }, [activeProjectId]);

  useEffect(() => {
    loadAssets();
  }, [loadAssets]);

  const highRiskRun = runs.some(
    (run) => run.report && (run.report.asr || run.report.metrics?.robustness?.attack_success_rate || 0) > 0.4,
  );
  const scopedHref = (href) =>
    activeProjectId
      ? `${href}${href.includes("?") ? "&" : "?"}project_id=${encodeURIComponent(activeProjectId)}`
      : href;

  return (
    <div className="space-y-5 animate-fade-in">
      <Card title="AdverTest là gì?" subtitle="Nền tảng kiểm thử độ bền vững mô hình AI trước tấn công đối kháng">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-3xl space-y-3 text-sm text-slate-600">
            <p>
              AdverTest giúp bạn đo mô hình nhìn thấy gì khi bị tấn công: sinh biến thể đối kháng (nhiễu, thời tiết, che
              khuất…), so sánh hiệu năng trước/sau, duyệt từng ảnh rồi đóng gói dataset adversarial để huấn luyện lại và
              xác nhận phòng thủ — toàn bộ trong một quy trình có báo cáo.
            </p>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="font-semibold text-slate-700">Bài toán hỗ trợ:</span>
              <span className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-emerald-700">
                Object Detection (2D)
              </span>
              <span className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-emerald-700">
                Segmentation
              </span>
              <span className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-emerald-700">
                3D Detection
              </span>
              <span className="rounded-md border border-slate-200 bg-slate-100 px-2 py-1 text-slate-500">
                Classification — sắp ra mắt
              </span>
            </div>
            <ol className="list-decimal space-y-0.5 pl-4 text-xs text-slate-500">
              <li>Tạo project, nạp mô hình &amp; dữ liệu gốc</li>
              <li>Chọn attack và mức severity, chạy benchmark</li>
              <li>Xem kết quả, duyệt ảnh, xuất adversarial dataset</li>
              <li>Huấn luyện lại &amp; đánh giá phòng thủ, xuất báo cáo</li>
            </ol>
          </div>
          <Link
            href={scopedHref("/experiments/new")}
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700"
          >
            <Workflow className="h-4 w-4" />
            Bắt đầu làm việc
          </Link>
        </div>
      </Card>
      <div className="grid grid-cols-1 gap-4">
        <Card title="Thao tác nhanh" subtitle="Lối tắt hành động thường dùng">
          <div className="grid grid-cols-2 gap-2">
            <Link
              href={scopedHref("/experiments/new")}
              className="rounded-lg border border-blue-200 bg-blue-50 p-2.5 text-center text-xs font-semibold text-blue-700 hover:bg-blue-100"
            >
              <Plus className="mx-auto mb-1 h-4 w-4" />
              Tạo thí nghiệm
            </Link>
            <button
              type="button"
              onClick={() => setUploadKind("model")}
              className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-center text-xs font-semibold text-slate-700 hover:bg-slate-100"
            >
              <Upload className="mx-auto mb-1 h-4 w-4" />
              Nạp mô hình
            </button>
            <button
              type="button"
              onClick={() => setUploadKind("dataset")}
              className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-center text-xs font-semibold text-slate-700 hover:bg-slate-100"
            >
              <CloudUpload className="mx-auto mb-1 h-4 w-4" />
              Nạp dataset
            </button>
            <Link
              href={scopedHref("/analysis")}
              className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-center text-xs font-semibold text-slate-700 hover:bg-slate-100"
            >
              <FileText className="mx-auto mb-1 h-4 w-4" />
              Xuất báo cáo
            </Link>
          </div>
          <Link href="/defense">
            <Button variant="outline" size="sm" className="mt-3 w-full justify-center text-xs">
              <Workflow className="mr-1.5 h-3.5 w-3.5 text-blue-600" />
              Quy trình phòng thủ đóng loop
            </Button>
          </Link>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card
          title="Mô hình đang khả dụng"
          subtitle="Checkpoint catalog đã đăng ký"
          headerAction={
            <Link href="/experiments/new" className="text-xs font-semibold text-blue-600">
              Cấu hình test →
            </Link>
          }
        >
          {models.length === 0 ? (
            <p className="py-5 text-center text-xs text-slate-500">
              {isLoading ? "Đang tải mô hình..." : "— / No verified data"}
            </p>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2">
              {models.map((model) => (
                <div
                  key={model.id}
                  className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50/70 px-3 py-2"
                >
                  <div className="min-w-0">
                    <p className="truncate text-xs font-semibold text-slate-800">{model.model_name || model.id}</p>
                    <p className="truncate font-mono text-[10px] text-slate-500">ID: {model.id}</p>
                    <p className="text-[10px] text-slate-500">{model.task}</p>
                  </div>
                  <Badge variant={model.runnable ? "success" : "secondary"}>
                    {model.runnable ? "Sẵn sàng" : "Chưa sẵn sàng"}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </Card>
        <Card
          title="Dataset đang khả dụng"
          subtitle="Các tập dữ liệu đã qua gate anonymization"
          headerAction={
            <Link href="/experiments/new" className="text-xs font-semibold text-blue-600">
              Chọn dataset →
            </Link>
          }
        >
          {datasets.length === 0 ? (
            <p className="py-5 text-center text-xs text-slate-500">
              {isLoading ? "Đang tải dataset..." : "— / No verified data"}
            </p>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2">
              {datasets.map((dataset) => (
                <div
                  key={dataset.name}
                  className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50/70 px-3 py-2"
                >
                  <div className="min-w-0">
                    <p className="truncate text-xs font-semibold text-slate-800">{dataset.title || dataset.name}</p>
                    <p className="text-[10px] text-slate-500">
                      {dataset.task_id} · {dataset.modality}
                    </p>
                  </div>
                  <Badge variant={dataset.anonymized ? "success" : "secondary"}>
                    {dataset.anonymized ? "Anonymized" : "Pending"}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <Card title="Cảnh báo & Khuyến nghị" subtitle="Tín hiệu chỉ dựa trên các phiên đã ghi nhận">
        {highRiskRun ? (
          <div className="flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50/60 p-3">
            <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-500" />
            <div>
              <p className="text-xs font-semibold text-slate-800">Suy giảm hiệu năng nghiêm trọng</p>
              <p className="mt-1 text-[11px] leading-snug text-slate-600">
                Phát hiện phiên thử nghiệm có ASR vượt ngưỡng 40%. Đề xuất đưa vào Retraining Backlog.
              </p>
            </div>
          </div>
        ) : (
          <div className="py-6 text-center text-xs font-medium text-slate-400">Chưa có cảnh báo nào</div>
        )}
      </Card>

      <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-4 py-3 text-xs text-slate-500">
        <span>
          {attacks.length ? `${attacks.length} attack methods trong catalog` : "Attack catalog chưa có dữ liệu"}
        </span>
        <Link href="/benchmark" className="font-semibold text-blue-600 hover:text-blue-800">
          Mở benchmark →
        </Link>
      </div>

      {uploadKind && (
        <QuickUploadDialog
          kind={uploadKind}
          project={activeProject}
          onClose={() => setUploadKind(null)}
          onSuccess={loadAssets}
        />
      )}
    </div>
  );
}
