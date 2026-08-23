"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  FileText,
  Sparkles,
  Download,
  Calendar,
  Filter,
  Share2,
  Table,
  CheckCircle,
  AlertTriangle,
  Lightbulb,
  Layers,
  BarChart2,
  FileSpreadsheet,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import DonutChart from "@/components/metrics/DonutChart";
import { ATTACK_RANKING_DATA } from "@/data/mockData";
import { cn } from "@/lib/utils";

const HEATMAP_MODELS = ["YOLOv8s", "YOLOv8n", "ResNet50", "ViT-B/16", "DeepLabV3+"];
const HEATMAP_ATTACKS = ["PGD", "FGSM", "AutoAttack", "CW", "Square", "DeepFool", "One-Pixel"];

const HEATMAP_MATRIX = {
  YOLOv8s: [58.4, 42.1, 49.2, 34.1, 24.5, 12.8, 6.2],
  YOLOv8n: [54.2, 38.6, 45.0, 31.8, 22.1, 10.5, 5.8],
  ResNet50: [44.6, 28.4, 32.1, 22.5, 16.4, 8.2, 3.4],
  "ViT-B/16": [38.2, 22.5, 26.8, 18.0, 12.5, 6.4, 2.1],
  "DeepLabV3+": [61.8, 46.2, 52.4, 38.0, 28.6, 14.2, 7.5],
};

const COMBINATIONS = [
  { name: "PGD → AutoAttack", asr: 51.42, color: "bg-red-500" },
  { name: "PGD → CW (L2)", asr: 46.17, color: "bg-red-400" },
  { name: "AutoAttack → Square", asr: 33.81, color: "bg-amber-500" },
  { name: "FGSM → PGD", asr: 31.24, color: "bg-amber-400" },
  { name: "PGD → Square Attack", asr: 28.96, color: "bg-blue-500" },
];

export default function AnalysisPage() {
  const [generatingReport, setGeneratingReport] = useState(false);
  const [reportGenerated, setReportGenerated] = useState(false);

  const handleGenerateAIReport = () => {
    setGeneratingReport(true);
    setTimeout(() => {
      setGeneratingReport(false);
      setReportGenerated(true);
    }, 1000);
  };

  const getHeatmapColor = (val) => {
    if (val >= 50) return "bg-red-500 text-white font-bold";
    if (val >= 35) return "bg-red-400 text-white font-bold";
    if (val >= 25) return "bg-amber-300 text-slate-900";
    if (val >= 15) return "bg-blue-100 text-slate-800";
    return "bg-slate-50 text-slate-600";
  };

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Phân tích & Báo cáo"
        subtitle="Phân tích chuyên sâu hiệu quả tấn công đối kháng, điểm yếu mô hình và hiệu quả phòng thủ."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Phân tích & Báo cáo" },
        ]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="primary"
              size="sm"
              icon={Sparkles}
              disabled={generatingReport}
              onClick={handleGenerateAIReport}
              className="bg-purple-600 hover:bg-purple-700 border-purple-600"
            >
              {generatingReport ? "Đang sinh báo cáo AI..." : "Tạo báo cáo tự động ✨"}
            </Button>
          </div>
        }
      />

      {/* 4 INSIGHT SUMMARY CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        <div className="p-3 rounded-lg bg-red-50 border border-red-200 space-y-1">
          <div className="text-[11px] font-bold text-red-700 uppercase">Tấn công mạnh nhất</div>
          <div className="text-sm font-bold text-red-900">PGD (ε=8/255)</div>
          <p className="text-xs text-red-700">ASR trung bình cao nhất đạt 42.16% trên toàn bộ tập benchmark.</p>
        </div>

        <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 space-y-1">
          <div className="text-[11px] font-bold text-amber-700 uppercase">Mô hình dễ tổn thương nhất</div>
          <div className="text-sm font-bold text-amber-900">YOLOv8s & YOLOv8n</div>
          <p className="text-xs text-amber-700">Nhóm YOLO 1-stage có tỷ lệ sụt giảm mAP cao hơn Transformer & ResNet.</p>
        </div>

        <div className="p-3 rounded-lg bg-blue-50 border border-blue-200 space-y-1">
          <div className="text-[11px] font-bold text-blue-700 uppercase">Độ nhạy theo cường độ</div>
          <div className="text-sm font-bold text-blue-900">Tăng gần tuyến tính</div>
          <p className="text-xs text-blue-700">Khi ε tăng từ 2/255 lên 8/255, ASR tăng trung bình 3.8% cho mỗi 1/255.</p>
        </div>

        <div className="p-3 rounded-lg bg-purple-50 border border-purple-200 space-y-1">
          <div className="text-[11px] font-bold text-purple-700 uppercase">Tổ hợp nguy hiểm nhất</div>
          <div className="text-sm font-bold text-purple-900">PGD → AutoAttack</div>
          <p className="text-xs text-purple-700">Kết hợp 2 giai đoạn làm ASR tăng thêm 9.3 điểm phần trăm.</p>
        </div>
      </div>

      {/* ATTACK RANKING TABLE & ATTACK TYPE DISTRIBUTION */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* Attack Ranking Table (8 Columns) */}
        <Card
          className="xl:col-span-8"
          title="Bảng xếp hạng tấn công (Theo ASR giảm dần)"
          subtitle="Đánh giá mức độ tàn phá của từng thuật toán đối kháng"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50">
                  <th className="py-2.5 px-3">#</th>
                  <th className="py-2.5 px-3">Loại tấn công</th>
                  <th className="py-2.5 px-3">ASR</th>
                  <th className="py-2.5 px-3">mAP giảm</th>
                  <th className="py-2.5 px-3">Mức ảnh hưởng</th>
                  <th className="py-2.5 px-3">Nhận xét</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                {ATTACK_RANKING_DATA.map((item) => (
                  <tr key={item.rank} className="hover:bg-slate-50">
                    <td className="py-2.5 px-3 font-bold text-slate-400">{item.rank}</td>
                    <td className="py-2.5 px-3 font-bold text-slate-900 font-mono">{item.attack}</td>
                    <td className="py-2.5 px-3 font-bold text-red-600">{item.asr}</td>
                    <td className="py-2.5 px-3 text-slate-600 font-mono">{item.mapDrop}</td>
                    <td className="py-2.5 px-3">
                      <Badge
                        variant={
                          item.severity === "Rất cao"
                            ? "danger"
                            : item.severity === "Cao"
                            ? "warning"
                            : item.severity === "Trung bình"
                            ? "primary"
                            : "success"
                        }
                      >
                        {item.severity}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-3 text-slate-500 text-[11px]">{item.remark}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Attack Type Distribution Donut (4 Columns) */}
        <Card
          className="xl:col-span-4"
          title="Phân bố các dạng tấn công"
          subtitle="Tỷ trọng theo cơ chế sinh nhiễu"
        >
          <DonutChart
            data={[
              { name: "Gradient-based", value: 59.4, color: "#2563EB" },
              { name: "Decision-based", value: 18.4, color: "#7C3AED" },
              { name: "Score-based", value: 14.6, color: "#0EA5A8" },
              { name: "Query-based", value: 7.6, color: "#F59E0B" },
            ]}
            centerValue="48"
            centerLabel="Phương pháp"
            height={220}
          />
        </Card>
      </div>

      {/* HEATMAP MATRIX & ATTACK COMBINATION */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* Heatmap Matrix (7 Columns) */}
        <Card
          className="xl:col-span-7"
          title="Heatmap ASR (%) theo Mô hình & Tấn công"
          subtitle="Ma trận tương quan độ nhạy cảm của các kiến trúc thị giác"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-center text-xs border-collapse font-mono">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-sans">
                  <th className="py-2 px-2 text-left">Mô hình</th>
                  {HEATMAP_ATTACKS.map((atk) => (
                    <th key={atk} className="py-2 px-1 text-[11px] font-semibold">{atk}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {HEATMAP_MODELS.map((model) => (
                  <tr key={model}>
                    <td className="py-2 px-2 text-left font-sans font-bold text-slate-800">{model}</td>
                    {HEATMAP_MATRIX[model].map((val, i) => (
                      <td key={i} className="py-1.5 px-1">
                        <div className={cn("py-1 rounded text-[10px]", getHeatmapColor(val))}>
                          {val}%
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-end gap-3 text-[10px] text-slate-500 mt-2">
            <span>Màu sắc ASR:</span>
            <span className="px-1.5 py-0.5 rounded bg-slate-100">&lt;15% Thấp</span>
            <span className="px-1.5 py-0.5 rounded bg-amber-200 text-slate-900">25-35% TB</span>
            <span className="px-1.5 py-0.5 rounded bg-red-500 text-white font-bold">&gt;50% Rất cao</span>
          </div>
        </Card>

        {/* Attack Combinations & Model Weakness (5 Columns) */}
        <Card
          className="xl:col-span-5"
          title="So sánh chuỗi tổ hợp tấn công"
          subtitle="Tác động cộng hưởng của các đợt tấn công nhiều giai đoạn"
        >
          <div className="space-y-3 text-xs">
            {COMBINATIONS.map((c) => (
              <div key={c.name} className="space-y-1">
                <div className="flex justify-between font-medium">
                  <span className="text-slate-800">{c.name}</span>
                  <span className="font-bold text-red-600 font-mono">{c.asr}%</span>
                </div>
                <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div className={cn("h-full rounded-full", c.color)} style={{ width: `${c.asr}%` }} />
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* REPORT PREVIEW CARD */}
      <Card
        title="Xem trước báo cáo tường thuật hoàn chỉnh"
        subtitle="Tổng hợp phân tích tự động chuẩn báo cáo khoa học & kỹ thuật"
        headerAction={
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" icon={FileSpreadsheet}>
              Excel
            </Button>
            <Button variant="secondary" size="sm" icon={Download}>
              Tải PDF
            </Button>
            <Button variant="secondary" size="sm" icon={Share2}>
              Chia sẻ
            </Button>
          </div>
        }
      >
        <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 space-y-4 font-normal leading-relaxed">
          <div>
            <h4 className="text-sm font-bold text-slate-900 mb-1">1. Tổng quan thí nghiệm</h4>
            <p>
              Trong tuần qua, hệ thống AdversAI Lab đã ghi nhận và phân tích 128 thí nghiệm đánh giá độ bền vững đối kháng trên 6 họ mô hình thị giác (YOLO, Faster R-CNN, RT-DETR, SAM, PointPillars, ViT). Dữ liệu kiểm thử bao gồm các kịch bản thời tiết, biến dạng quang học và 48 phương pháp tấn công đối kháng chuẩn quốc tế.
            </p>
          </div>

          <div>
            <h4 className="text-sm font-bold text-slate-900 mb-1">2. Phát hiện & Điểm yếu cốt lõi</h4>
            <ul className="list-disc pl-4 space-y-1 text-slate-700">
              <li><strong>PGD và AutoAttack</strong> là hai thuật toán có sức tàn phá mạnh nhất, khiến độ chính xác mAP@0.5 sụt giảm trung bình 68.7% trên các mô hình 1-stage YOLO.</li>
              <li><strong>ViT-B/16 và ResNet50</strong> thể hiện khả năng duy trì đặc trưng ngữ nghĩa tốt hơn, tỷ lệ ASR thấp hơn 16-20% so với YOLOv8n ở cùng mức nhiễu ε=8/255.</li>
              <li><strong>Tổ hợp PGD → AutoAttack</strong> gây ra hiện tượng gradient masking sụp đổ, khiến mô hình hoàn toàn mất khả năng phát hiện vật thể.</li>
            </ul>
          </div>

          <div>
            <h4 className="text-sm font-bold text-slate-900 mb-1">3. Khuyến nghị & Lộ trình phòng thủ</h4>
            <ul className="list-disc pl-4 space-y-1 text-slate-700">
              <li>Khởi chạy quy trình <strong>Adversarial Training (PGD-AT với TRADES Loss)</strong> ngay trong chu kỳ fine-tuning tiếp theo.</li>
              <li>Tích hợp tầng tiền xử lý <strong>Randomized Smoothing và Feature Denoising</strong> tại inference pipeline để giảm thiểu độ nhạy của gradient.</li>
              <li>Theo dõi chặt chẽ và lưu vết các vector tấn công vật lý (Adversarial Patch) khi triển khai tại môi trường xe tự hành thực địa.</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}
