import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Gaia — Climate Explorer',
  description: 'Visualize 70 years of ERA5 climate data with ML-powered anomaly detection.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}