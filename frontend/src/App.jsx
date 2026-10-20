import { useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'
import { demoData, sentimentTrend, mapInputToModelSchema, calculateOpportunitySummary } from './api'

const initialForm = {
  productName: 'AeroBass X9',
  category: 'TWS Earbuds',
  discountedPrice: 2499,
  actualPrice: 3499,
  targetPriceMin: 1999,
  targetPriceMax: 2999,
  rating: 4.3,
  ratingVelocity: 28,
  marketChannel: 'Amazon (Online)',
  hardwareScore: 82,
  sentimentScore: 76,
  competitorCount: 42,
  isWireless: true,
  hasAnc: true,
  hasEnc: true,
  hasMic: true,
  hasFastCharge: true,
  hasLowLatency: false,
  hasVoiceAssistant: true,
  bluetoothVersion: 5.3,
  ipxRating: 4,
  driverMm: 12,
  playtimeHrs: 30,
  targetMargin: 22
}

const badgeClasses = {
  High: 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30',
  Medium: 'bg-amber-500/15 text-amber-200 ring-1 ring-amber-500/30',
  Low: 'bg-rose-500/15 text-rose-200 ring-1 ring-rose-500/30'
}

function App() {
  const [form, setForm] = useState(initialForm)

  const schema = useMemo(() => mapInputToModelSchema(form), [form])
  const summary = useMemo(() => calculateOpportunitySummary(schema), [schema])

  const cards = [
    {
      title: '30-day demand growth',
      value: `${schema.demand_growth_projection.toFixed(1)}%`,
      badge: summary.level,
      note: 'Projected demand pull from model output'
    },
    {
      title: 'Customer sentiment',
      value: `${summary.sentiment.toFixed(0)}%`,
      badge: summary.sentiment > 70 ? 'High' : summary.sentiment > 45 ? 'Medium' : 'Low',
      note: 'Positive feedback share from VADER score'
    },
    {
      title: 'Price band',
      value: `₹${form.targetPriceMin.toLocaleString()} - ₹${form.targetPriceMax.toLocaleString()}`,
      badge: 'Medium',
      note: 'Competitive pricing window'
    },
    {
      title: 'Market crowding',
      value: `${schema.competitor_count} products`,
      badge: schema.competitor_count > 50 ? 'High' : schema.competitor_count > 25 ? 'Medium' : 'Low',
      note: 'Competitive intensity in active segment'
    }
  ]

  const setField = (field, value) => setForm((current) => ({ ...current, [field]: value }))

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8 flex flex-col gap-5 rounded-3xl border border-white/10 bg-white/5 p-6 shadow-soft backdrop-blur-sm">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-violet-300">MLOps market intelligence</p>
              <h1 className="mt-2 text-3xl font-bold text-white">Consumer Electronics Opportunity Analyzer</h1>
            </div>
            <div className="flex items-center gap-3 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-200">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" />
              XGBoost demand model active
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-4">
            {cards.map((card) => (
              <div key={card.title} className="rounded-2xl border border-white/10 bg-slate-900/60 p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <p className="text-sm text-slate-300">{card.title}</p>
                  <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide ${badgeClasses[card.badge] || 'bg-slate-500/15 text-slate-200 ring-1 ring-slate-500/25'}`}>
                    {card.badge}
                  </span>
                </div>
                <div className="text-2xl font-bold text-white">{card.value}</div>
                <p className="mt-2 text-xs text-slate-400">{card.note}</p>
              </div>
            ))}
          </div>
        </header>

        <main className="grid gap-8 xl:grid-cols-[400px_minmax(0,1fr)]">
          <aside className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-soft">
            <div className="mb-5 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Product inputs</h2>
              <span className="rounded-full bg-violet-500/10 px-2 py-1 text-xs text-violet-200">Live schema</span>
            </div>

            <div className="space-y-5">
              <div>
                <label className="mb-2 block text-sm text-slate-300">Product name</label>
                <input value={form.productName} onChange={(e) => setField('productName', e.target.value)} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none ring-0 transition focus:border-violet-400" />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-2 block text-sm text-slate-300">Category</label>
                  <select value={form.category} onChange={(e) => setField('category', e.target.value)} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none focus:border-violet-400">
                    <option>TWS Earbuds</option>
                    <option>Neckband</option>
                    <option>Wireless Headphones</option>
                    <option>Smart Speaker</option>
                  </select>
                </div>
                <div>
                  <label className="mb-2 block text-sm text-slate-300">Channel</label>
                  <select value={form.marketChannel} onChange={(e) => setField('marketChannel', e.target.value)} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none focus:border-violet-400">
                    <option>Amazon (Online)</option>
                    <option>Flipkart (Online)</option>
                    <option>Croma (Offline Retail Chain)</option>
                    <option>Offline Electronics Market (Dealer MOP)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-2 block text-sm text-slate-300">Target price min</label>
                  <input type="number" value={form.targetPriceMin} onChange={(e) => setField('targetPriceMin', Number(e.target.value))} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none focus:border-violet-400" />
                </div>
                <div>
                  <label className="mb-2 block text-sm text-slate-300">Target price max</label>
                  <input type="number" value={form.targetPriceMax} onChange={(e) => setField('targetPriceMax', Number(e.target.value))} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none focus:border-violet-400" />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-2 block text-sm text-slate-300">List price</label>
                  <input type="number" value={form.actualPrice} onChange={(e) => setField('actualPrice', Number(e.target.value))} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none focus:border-violet-400" />
                </div>
                <div>
                  <label className="mb-2 block text-sm text-slate-300">Selling price</label>
                  <input type="number" value={form.discountedPrice} onChange={(e) => setField('discountedPrice', Number(e.target.value))} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none focus:border-violet-400" />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-2 block text-sm text-slate-300">Rating</label>
                  <input type="number" step="0.1" min="1" max="5" value={form.rating} onChange={(e) => setField('rating', Number(e.target.value))} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none focus:border-violet-400" />
                </div>
                <div>
                  <label className="mb-2 block text-sm text-slate-300">Rating velocity</label>
                  <input type="number" value={form.ratingVelocity} onChange={(e) => setField('ratingVelocity', Number(e.target.value))} className="w-full rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none focus:border-violet-400" />
                </div>
              </div>

              <div>
                <label className="mb-2 block text-sm text-slate-300">Hardware score</label>
                <input type="range" min="0" max="100" value={form.hardwareScore} onChange={(e) => setField('hardwareScore', Number(e.target.value))} className="w-full" />
                <div className="mt-1 text-right text-xs text-slate-400">{form.hardwareScore}/100</div>
              </div>

              <div>
                <label className="mb-2 block text-sm text-slate-300">Sentiment score</label>
                <input type="range" min="0" max="100" value={form.sentimentScore} onChange={(e) => setField('sentimentScore', Number(e.target.value))} className="w-full" />
                <div className="mt-1 text-right text-xs text-slate-400">{form.sentimentScore}/100</div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  ['isWireless', 'Wireless'],
                  ['hasAnc', 'ANC'],
                  ['hasEnc', 'ENC'],
                  ['hasMic', 'Mic'],
                  ['hasFastCharge', 'Fast charge'],
                  ['hasLowLatency', 'Low latency'],
                  ['hasVoiceAssistant', 'Voice assistant']
                ].map(([key, label]) => (
                  <label key={key} className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-950/50 px-3 py-2 text-sm text-slate-200">
                    <span>{label}</span>
                    <input type="checkbox" checked={form[key]} onChange={(e) => setField(key, e.target.checked)} className="h-4 w-4 rounded border-slate-600 bg-slate-900 text-violet-500 focus:ring-violet-500" />
                  </label>
                ))}
              </div>
            </div>
          </aside>

          <section className="space-y-8">
            <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-soft">
              <div className="mb-5 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-white">Market opportunity summary</h2>
                <span className={`rounded-full px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] ${badgeClasses[summary.level]}`}>
                  {summary.level}
                </span>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-2xl border border-violet-500/20 bg-violet-500/10 p-4">
                  <p className="text-sm text-violet-200">Opportunity score</p>
                  <p className="mt-2 text-3xl font-bold text-white">{summary.level}</p>
                  <p className="mt-2 text-sm text-slate-300">{summary.signal}</p>
                </div>
                <div className="rounded-2xl border border-cyan-500/20 bg-cyan-500/10 p-4">
                  <p className="text-sm text-cyan-200">Projected growth</p>
                  <p className="mt-2 text-3xl font-bold text-white">+{schema.demand_growth_projection.toFixed(1)}%</p>
                  <p className="mt-2 text-sm text-slate-300">30-day review-count demand proxy</p>
                </div>
                <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/10 p-4">
                  <p className="text-sm text-emerald-200">Sentiment momentum</p>
                  <p className="mt-2 text-3xl font-bold text-white">{summary.sentiment.toFixed(0)}%</p>
                  <p className="mt-2 text-sm text-slate-300">Positive customer response at market level</p>
                </div>
              </div>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                  <p className="text-sm text-slate-400">Model input schema</p>
                  <pre className="mt-3 overflow-auto text-xs text-slate-200">{JSON.stringify(schema, null, 2)}</pre>
                </div>
                <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
                  <p className="text-sm text-slate-400">Business assessment</p>
                  <ul className="mt-3 space-y-2 text-sm text-slate-300">
                    <li>• Recommended price band: ₹{form.targetPriceMin.toLocaleString()}–₹{form.targetPriceMax.toLocaleString()}</li>
                    <li>• Customer sentiment: {summary.sentiment.toFixed(0)}% positive</li>
                    <li>• Feature pull: {form.hardwareScore}/100 hardware suitability</li>
                    <li>• Competitive intensity: {schema.competitor_count} visible products</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="grid gap-8 lg:grid-cols-2">
              <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-soft">
                <h3 className="mb-4 text-lg font-semibold text-white">Demand growth trend</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={demoData}>
                      <defs>
                        <linearGradient id="growthFill" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.6} />
                          <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="month" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip />
                      <Area type="monotone" dataKey="growth" stroke="#8b5cf6" strokeWidth={3} fill="url(#growthFill)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-900/70 p-5 shadow-soft">
                <h3 className="mb-4 text-lg font-semibold text-white">Customer sentiment</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={sentimentTrend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="day" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip />
                      <Bar dataKey="positive" fill="#34d399" radius={[8, 8, 0, 0]} />
                      <Bar dataKey="neutral" fill="#fbbf24" radius={[8, 8, 0, 0]} />
                      <Bar dataKey="negative" fill="#f87171" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}

export default App
