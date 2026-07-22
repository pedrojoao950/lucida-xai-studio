import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://explainable-ai-studio.pedrojoao950.chatgpt.site"),
  title: "LÚCIDA — Observatório de Decisões",
  description: "Entre na mente do modelo. Um observatório vivo para compreender, questionar e transformar decisões de inteligência artificial.",
  icons: { icon: "/favicon.svg" },
  openGraph: {
    title: "LÚCIDA — Observatório de Decisões",
    description: "Entre na mente do modelo.",
    images: [{ url: "/og.png", width: 1536, height: 1024, alt: "LÚCIDA — Observatório de Decisões" }],
  },
  twitter: { card: "summary_large_image", images: ["/og.png"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="pt"><body>{children}</body></html>;
}
