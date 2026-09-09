import { Inter } from "next/font/google";
import Script from "next/script";
import AppProviders from "@/components/AppProviders";
import AppShell from "@/components/layout/AppShell";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "AdverTest — Nền tảng đánh giá & phòng thủ AI đối kháng",
  description: "Nền tảng kiểm thử độ bền vững và huấn luyện phòng thủ mô hình AI thị giác chuyên sâu.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="vi" className={inter.className} suppressHydrationWarning>
      <head>
        <Script src="/runtime-config.js" strategy="beforeInteractive" />
      </head>
      <body className="antialiased bg-[var(--app-bg)] text-[var(--app-text)]">
        <AppProviders>
          <AppShell>{children}</AppShell>
        </AppProviders>
      </body>
    </html>
  );
}
