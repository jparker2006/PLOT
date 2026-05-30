import "./globals.css";

export const metadata = {
  title: "PLOT — Points Left On the Table",
  description: "chess.com's post-game review, but for NBA possessions: an eval bar + per-decision regret.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
