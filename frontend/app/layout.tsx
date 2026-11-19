import type { Metadata } from "next";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import "./globals.css";

export const metadata: Metadata = {
  title: "Component-Level RAG & Agentic Assurance Analyzer",
  description: "Modular observability and substantive evaluation for enterprise RAG pipelines.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="min-h-screen bg-zinc-50 font-sans text-zinc-950 antialiased">{children}</body>
    </html>
  );
}
