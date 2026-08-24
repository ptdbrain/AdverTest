"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Users,
  ShieldAlert,
  Database,
  Layers,
  Activity,
  CheckCircle,
  Clock,
  Search,
  Plus,
  Edit,
  Eye,
  MoreVertical,
  Key,
  Bell,
  HardDrive,
  Cpu,
  RefreshCw,
  Download,
  Settings,
  ShieldCheck,
  Check,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import DonutChart from "@/components/metrics/DonutChart";
import {
  ADMIN_USERS,
  ADMIN_APPROVALS,
  SYSTEM_SERVICES,
  AUDIT_LOGS,
} from "@/data/mockData";
import { cn } from "@/lib/utils";

const ADMIN_SUBTABS = [
  { id: "users", label: "Người dùng & Phân quyền" },
  { id: "repositories", label: "Kho tài nguyên" },
  { id: "approvals", label: "Quy trình duyệt" },
  { id: "monitoring", label: "Giám sát hệ thống" },
  { id: "api", label: "API & Tích hợp" },
  { id: "audit", label: "Lịch sử & Nhật ký" },
];

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState("users");
  const [userSearch, setUserSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");

  const filteredUsers = ADMIN_USERS.filter((u) => {
    const matchSearch =
      u.name.toLowerCase().includes(userSearch.toLowerCase()) ||
      u.email.toLowerCase().includes(userSearch.toLowerCase());
    const matchRole = roleFilter === "all" || u.role === roleFilter;
    return matchSearch && matchRole;
  });

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Quản trị hệ thống"
        subtitle="Quản lý người dùng, tài nguyên, phân quyền chính sách và giám sát hoạt động hệ thống"
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Quản trị hệ thống" },
        ]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" icon={RefreshCw}>
              Làm mới
            </Button>
            <Button variant="secondary" size="sm" icon={Download}>
              Xuất báo cáo
            </Button>
            <Button variant="primary" size="sm" icon={Settings}>
              Cấu hình hệ thống
            </Button>
          </div>
        }
      />

      {/* 5 ADMIN KPI CARDS */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
        <div className="p-3 rounded-lg bg-white border border-slate-200 shadow-xs">
          <div className="text-[11px] text-slate-500 font-medium">Người dùng hoạt động</div>
          <div className="text-xl font-bold text-slate-900 mt-1">128</div>
          <div className="text-[10px] text-emerald-600 font-semibold mt-1">↑ 18% tuần này</div>
        </div>

        <div className="p-3 rounded-lg bg-white border border-slate-200 shadow-xs">
          <div className="text-[11px] text-slate-500 font-medium">Thí nghiệm (7 ngày)</div>
          <div className="text-xl font-bold text-slate-900 mt-1">342</div>
          <div className="text-[10px] text-blue-600 font-semibold mt-1">Đang xử lý 3 luồng</div>
        </div>

        <div className="p-3 rounded-lg bg-white border border-slate-200 shadow-xs">
          <div className="text-[11px] text-slate-500 font-medium">Dung lượng sử dụng</div>
          <div className="text-xl font-bold text-slate-900 mt-1">1.52 TB <span className="text-xs text-slate-400 font-normal">/ 5 TB</span></div>
          <div className="text-[10px] text-slate-500 mt-1">30% quota lưu trữ</div>
        </div>

        <div className="p-3 rounded-lg bg-white border border-slate-200 shadow-xs">
          <div className="text-[11px] text-slate-500 font-medium">Tình trạng hệ thống</div>
          <div className="text-xl font-bold text-emerald-600 mt-1">Khỏe mạnh</div>
          <div className="text-[10px] text-slate-500 mt-1">99.9% uptime 5 cụm dịch vụ</div>
        </div>

        <div className="p-3 rounded-lg bg-white border border-slate-200 shadow-xs">
          <div className="text-[11px] text-slate-500 font-medium">Yêu cầu duyệt chờ</div>
          <div className="text-xl font-bold text-amber-600 mt-1">5</div>
          <div className="text-[10px] text-amber-700 font-semibold mt-1">Cần xử lý phê duyệt</div>
        </div>
      </div>

      {/* ADMIN SUBTABS BAR */}
      <div className="flex items-center gap-1 overflow-x-auto border-b border-slate-200 pb-1">
        {ADMIN_SUBTABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap",
              activeTab === tab.id
                ? "bg-blue-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* TAB CONTENT: USER & PERMISSIONS */}
      {activeTab === "users" && (
        <div className="space-y-5">
          {/* Roles Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3.5 rounded-lg bg-white border border-slate-200 shadow-xs space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-slate-800">Người dùng (User)</span>
                <Badge variant="default">76 users</Badge>
              </div>
              <p className="text-[11px] text-slate-500">
                Truy cập nền tảng, chạy các bài đánh giá với tài nguyên được cấp riêng biệt.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-white border border-slate-200 shadow-xs space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-blue-700">Nghiên cứu viên (Researcher)</span>
                <Badge variant="primary">38 users</Badge>
              </div>
              <p className="text-[11px] text-slate-500">
                Tạo thí nghiệm, tùy biến tham số tấn công và huấn luyện phòng thủ chuyên sâu.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-white border border-slate-200 shadow-xs space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-purple-700">Admin (Quản trị viên)</span>
                <Badge variant="purple">14 users</Badge>
              </div>
              <p className="text-[11px] text-slate-500">
                Quản trị toàn diện hệ thống, duyệt checkpoints/datasets và theo dõi audit logs.
              </p>
            </div>
          </div>

          {/* User Management Table */}
          <Card
            title="Danh sách người dùng & Phân bổ tài nguyên"
            subtitle="Quản lý hạn ngạch tính toán GPU và phân quyền tài khoản"
            headerAction={
              <Button variant="primary" size="sm" icon={Plus}>
                Tạo người dùng mới
              </Button>
            }
          >
            {/* Filter & Search Toolbar */}
            <div className="flex flex-col sm:flex-row gap-3 mb-4 text-xs">
              <div className="relative flex-1">
                <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
                <input
                  type="text"
                  placeholder="Tìm kiếm người dùng theo tên hoặc email..."
                  value={userSearch}
                  onChange={(e) => setUserSearch(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 focus:bg-white focus:outline-none focus:border-blue-500"
                />
              </div>
              <select
                value={roleFilter}
                onChange={(e) => setRoleFilter(e.target.value)}
                className="p-1.5 rounded-lg border border-slate-200 bg-white font-medium"
              >
                <option value="all">Tất cả vai trò</option>
                <option value="Admin">Admin</option>
                <option value="Nghiên cứu viên">Nghiên cứu viên</option>
                <option value="Người dùng">Người dùng</option>
              </select>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50">
                    <th className="py-2.5 px-3">Họ và tên</th>
                    <th className="py-2.5 px-3">Email</th>
                    <th className="py-2.5 px-3">Vai trò</th>
                    <th className="py-2.5 px-3">Trạng thái</th>
                    <th className="py-2.5 px-3">Sử dụng tài nguyên</th>
                    <th className="py-2.5 px-3 text-right">Thao tác</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                  {filteredUsers.map((user) => (
                    <tr key={user.id} className="hover:bg-slate-50">
                      <td className="py-2.5 px-3 font-bold text-slate-900">{user.name}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-500">{user.email}</td>
                      <td className="py-2.5 px-3">
                        <Badge
                          variant={
                            user.role === "Admin"
                              ? "purple"
                              : user.role === "Nghiên cứu viên"
                              ? "primary"
                              : "default"
                          }
                        >
                          {user.role}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3">
                        <Badge variant={user.status === "Hoạt động" ? "success" : "danger"} dot>
                          {user.status}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3">
                        <div className="flex items-center gap-2">
                          <div className="w-24 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                            <div
                              className={cn(
                                "h-full rounded-full",
                                user.quota >= 80 ? "bg-red-500" : "bg-blue-500"
                              )}
                              style={{ width: `${user.quota}%` }}
                            />
                          </div>
                          <span className="font-mono text-[11px]">{user.quota}%</span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <div className="flex items-center justify-end gap-1 text-slate-400">
                          <button type="button" className="p-1 hover:text-slate-700" title="Xem chi tiết">
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                          <button type="button" className="p-1 hover:text-slate-700" title="Chỉnh sửa">
                            <Edit className="w-3.5 h-3.5" />
                          </button>
                          <button type="button" className="p-1 hover:text-slate-700">
                            <MoreVertical className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* TAB CONTENT: RESOURCE REPOSITORIES */}
      {activeTab === "repositories" && (
        <div className="space-y-5">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            <Card title="Kho mô hình (Model Repo)" subtitle="Quản lý weights và checkpoints">
              <div className="space-y-2 text-xs divide-y divide-slate-100 font-medium">
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Tổng số mô hình</span>
                  <span className="font-bold text-slate-800">56</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Phiên bản checkpoints</span>
                  <span className="text-slate-800">128</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Tổng kích thước</span>
                  <span className="font-mono text-blue-600 font-bold">412 GB</span>
                </div>
              </div>
            </Card>

            <Card title="Kho dữ liệu (Dataset Repo)" subtitle="Dữ liệu kiểm thử & nhãn annotation">
              <div className="space-y-2 text-xs divide-y divide-slate-100 font-medium">
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Tổng số Dataset</span>
                  <span className="font-bold text-slate-800">34</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Số bản ghi</span>
                  <span className="text-slate-800">12.4M</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Tổng kích thước</span>
                  <span className="font-mono text-purple-600 font-bold">678 GB</span>
                </div>
              </div>
            </Card>

            <Card title="Thư viện tấn công" subtitle="Các module sinh nhiễu đối kháng">
              <div className="space-y-2 text-xs divide-y divide-slate-100 font-medium">
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Phương pháp</span>
                  <span className="font-bold text-slate-800">48</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Lượt thực thi</span>
                  <span className="text-slate-800">1.2K</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Thuật toán mới</span>
                  <Badge variant="danger">+3 mới</Badge>
                </div>
              </div>
            </Card>

            <Card title="Thư viện phòng thủ" subtitle="Các cơ chế bảo vệ & tôi luyện">
              <div className="space-y-2 text-xs divide-y divide-slate-100 font-medium">
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Kỹ thuật phòng thủ</span>
                  <span className="font-bold text-slate-800">28</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Lượt áp dụng</span>
                  <span className="text-slate-800">892</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Mô hình đã tôi luyện</span>
                  <Badge variant="success">14 models</Badge>
                </div>
              </div>
            </Card>
          </div>

          {/* Model Catalog Table */}
          <Card
            title="Danh mục mô hình sẵn có (Model Catalog)"
            subtitle="Danh sách các checkpoints và weights thực tế trong thư mục máy & chuẩn hệ thống"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50">
                    <th className="py-2.5 px-3">Tên mô hình & Checkpoint</th>
                    <th className="py-2.5 px-3">Nguồn</th>
                    <th className="py-2.5 px-3">Kiến trúc</th>
                    <th className="py-2.5 px-3">Nhiệm vụ</th>
                    <th className="py-2.5 px-3">Số tham số</th>
                    <th className="py-2.5 px-3">Kích thước</th>
                    <th className="py-2.5 px-3">Baseline</th>
                    <th className="py-2.5 px-3">FPS (RTX 4090)</th>
                    <th className="py-2.5 px-3 text-right">Hành động</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                  {AVAILABLE_MODELS.map((m) => (
                    <tr key={m.id} className={cn("hover:bg-slate-50", m.isLocal && "bg-blue-50/20")}>
                      <td className="py-2.5 px-3">
                        <div className="font-bold text-slate-900 font-mono flex items-center gap-1.5">
                          <span>{m.isLocal ? "📍" : "🤖"}</span>
                          <span>{m.name}</span>
                        </div>
                        {m.localPath && (
                          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                            {m.localPath}
                          </div>
                        )}
                      </td>
                      <td className="py-2.5 px-3">
                        {m.isLocal ? (
                          <Badge variant="success">Local Máy</Badge>
                        ) : (
                          <Badge variant="default">Pretrained</Badge>
                        )}
                      </td>
                      <td className="py-2.5 px-3">
                        <Badge variant="primary">{m.family}</Badge>
                      </td>
                      <td className="py-2.5 px-3 text-slate-600">{m.task}</td>
                      <td className="py-2.5 px-3 font-mono font-bold text-blue-600">{m.params}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-500">{m.fileSize}</td>
                      <td className="py-2.5 px-3 font-mono font-bold text-emerald-600">{m.mapBaseline}</td>
                      <td className="py-2.5 px-3 font-mono text-purple-600">{m.fpsRtx4090}</td>
                      <td className="py-2.5 px-3 text-right">
                        <Link href="/experiments/new">
                          <Button variant="secondary" size="sm" className="text-[11px] py-1 px-2">
                            Dùng mô hình
                          </Button>
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Dataset Catalog Table */}
          <Card
            title="Danh mục tập dữ liệu sẵn có (Dataset Catalog)"
            subtitle="Các bộ benchmark & dữ liệu thực tế lưu trữ trong workspace"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50">
                    <th className="py-2.5 px-3">Tên tập dữ liệu</th>
                    <th className="py-2.5 px-3">Nguồn</th>
                    <th className="py-2.5 px-3">Nhiệm vụ</th>
                    <th className="py-2.5 px-3">Số lượng mẫu</th>
                    <th className="py-2.5 px-3">Số lớp</th>
                    <th className="py-2.5 px-3">Dung lượng</th>
                    <th className="py-2.5 px-3">Định dạng</th>
                    <th className="py-2.5 px-3 text-right">Hành động</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                  {AVAILABLE_DATASETS.map((d) => (
                    <tr key={d.id} className={cn("hover:bg-slate-50", d.isLocal && "bg-emerald-50/20")}>
                      <td className="py-2.5 px-3">
                        <div className="font-bold text-slate-900 flex items-center gap-1.5">
                          <span>{d.isLocal ? "📍" : "📦"}</span>
                          <span>{d.name}</span>
                        </div>
                        {d.localPath && (
                          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                            {d.localPath}
                          </div>
                        )}
                      </td>
                      <td className="py-2.5 px-3">
                        {d.isLocal ? (
                          <Badge variant="success">Local Máy</Badge>
                        ) : (
                          <Badge variant="teal">Dataset Repo</Badge>
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-slate-600">{d.task}</td>
                      <td className="py-2.5 px-3 font-mono font-bold text-blue-600">{d.samples.toLocaleString()}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-700">{d.classes} lớp</td>
                      <td className="py-2.5 px-3 font-mono text-purple-600 font-bold">{d.size}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-500">{d.format}</td>
                      <td className="py-2.5 px-3 text-right">
                        <Link href="/experiments/new">
                          <Button variant="secondary" size="sm" className="text-[11px] py-1 px-2">
                            Chọn Dataset
                          </Button>
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* TAB CONTENT: APPROVALS */}
      {activeTab === "approvals" && (
        <Card
          title="Quy trình phê duyệt (Chờ xử lý)"
          subtitle="Kiểm duyệt các mô hình và tập dữ liệu do thành viên tải lên"
          headerAction={
            <Button variant="primary" size="sm" icon={Check}>
              Duyệt tất cả (5)
            </Button>
          }
        >
          <div className="space-y-3">
            {ADMIN_APPROVALS.map((app) => (
              <div
                key={app.id}
                className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-xs"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant={app.type === "Mô hình" ? "primary" : "teal"}>{app.type}</Badge>
                    <span className="font-bold text-slate-900">{app.name}</span>
                  </div>
                  <div className="text-slate-500 text-[11px]">
                    Tải lên bởi: <strong className="text-slate-700">{app.uploader}</strong> · {app.time}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="secondary" size="sm">
                    Xem kiểm thử
                  </Button>
                  <Button variant="success" size="sm">
                    Duyệt ngay
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* TAB CONTENT: SYSTEM MONITORING */}
      {activeTab === "monitoring" && (
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
          <Card
            className="xl:col-span-7"
            title="Sức khỏe các dịch vụ hệ thống"
            subtitle="Theo dõi thời gian hoạt động thực tế (SLA 99.9%)"
          >
            <div className="divide-y divide-slate-100 text-xs font-medium">
              {SYSTEM_SERVICES.map((srv) => (
                <div key={srv.name} className="py-2.5 flex items-center justify-between">
                  <div className="space-y-0.5">
                    <span className="font-bold text-slate-800">{srv.name}</span>
                    <div className="text-[10px] text-slate-400">Độ trễ trung bình: {srv.latency}</div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-emerald-600 font-bold">{srv.uptime}</span>
                    <Badge variant="success" dot>{srv.status}</Badge>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card
            className="xl:col-span-5"
            title="Phân bổ dung lượng lưu trữ"
            subtitle="Tổng dung lượng 1.52 TB / 5.0 TB"
          >
            <DonutChart
              data={[
                { name: "Mô hình (Weights)", value: 412, color: "#2563EB" },
                { name: "Dữ liệu (Datasets)", value: 678, color: "#7C3AED" },
                { name: "Kết quả thí nghiệm", value: 256, color: "#0EA5A8" },
                { name: "Khác", value: 176, color: "#94A3B8" },
              ]}
              centerValue="1.52 TB"
              centerLabel="Đã sử dụng (30%)"
              height={220}
            />
          </Card>
        </div>
      )}

      {/* TAB CONTENT: AUDIT LOGS */}
      {activeTab === "audit" && (
        <Card
          title="Lịch sử & Nhật ký hoạt động (Audit Logs)"
          subtitle="Ghi lại toàn bộ hành động thay đổi cấu hình và phê duyệt tài nguyên"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50">
                  <th className="py-2.5 px-3">Thời gian</th>
                  <th className="py-2.5 px-3">Người thực hiện</th>
                  <th className="py-2.5 px-3">Hành động</th>
                  <th className="py-2.5 px-3">Địa chỉ IP</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                {AUDIT_LOGS.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-50">
                    <td className="py-2.5 px-3 font-mono text-slate-500">{log.time}</td>
                    <td className="py-2.5 px-3 font-bold text-slate-900">{log.user}</td>
                    <td className="py-2.5 px-3 text-slate-800">{log.action}</td>
                    <td className="py-2.5 px-3 font-mono text-slate-400">{log.ip}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
