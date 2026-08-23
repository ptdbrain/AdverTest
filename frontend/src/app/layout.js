import { Inter } from "next/font/google";
import AppShell from "@/components/layout/AppShell";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "AdversAI Lab — Nền tảng đánh giá & phòng thủ AI đối kháng",
  description: "Nền tảng kiểm thử độ bền vững và huấn luyện phòng thủ mô hình AI thị giác chuyên sâu.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="vi" className={inter.className}>
      <body className="antialiased bg-[#F8FAFC] text-[#0F172A]">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
