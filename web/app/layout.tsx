import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Stocknet Dashboard",
  description: "Training progress and results for stocknet models",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <header className="border-b border-gray-800 px-6 py-4 flex items-center gap-8">
          <a href="/" className="text-xl font-semibold tracking-tight text-white hover:text-blue-400 transition-colors">
            Stocknet Dashboard
          </a>
          <nav className="flex gap-4 text-sm">
            <a href="/" className="text-gray-400 hover:text-white transition-colors">Models</a>
            <a href="/compare" className="text-gray-400 hover:text-white transition-colors">Compare</a>
            <a href="/datasources" className="text-gray-400 hover:text-white transition-colors">Data Sources</a>
          </nav>
        </header>
        <main className="px-6 py-8 max-w-7xl mx-auto">{children}</main>
      </body>
    </html>
  );
}
