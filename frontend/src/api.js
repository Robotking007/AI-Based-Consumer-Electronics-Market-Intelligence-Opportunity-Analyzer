const baseSchema = {
  product_name: 'AeroBass X9',
  category: 'TWS Earbuds',
  discounted_price: 2499,
  actual_price: 3499,
  discount_pct: 28,
  rating: 4.3,
  rating_count: 18520,
  market_channel: 'Amazon (Online)',
  is_wireless: 1,
  has_anc: 1,
  has_enc: 1,
  has_mic: 1,
  fast_charge: 1,
  low_latency: 0,
  voice_assistant: 1,
  bt_version: 5.3,
  ipx_rating: 4,
  driver_mm: 12,
  playtime_hrs: 30,
  sentiment_compound: 0.64,
  pct_positive_reviews: 0.76,
  competitor_count: 42,
  price_vs_median: 1.28,
  target_price_min: 1999,
  target_price_max: 2999,
  rating_velocity: 18,
  target_margin: 22,
  demand_growth_projection: 16.4,
  opportunity_level: 'High'
}

const clamp = (value, min, max) => Math.min(Math.max(value, min), max)

export function mapInputToModelSchema(form) {
  const discountPct = clamp(
    Math.round(((form.actualPrice - form.discountedPrice) / form.actualPrice) * 100),
    0,
    90
  )

  const ratingCount = Math.round(form.ratingVelocity * 2500 + form.targetPriceMax * 4.5)
  const sentiment = clamp(form.sentimentScore / 100, 0.1, 0.95)
  const growth = clamp(
    Number(
      (
        (form.hardwareScore * 0.22 +
          form.ratingVelocity * 0.26 +
          form.sentimentScore * 0.2 -
          (Math.abs(form.discountedPrice - ((form.targetPriceMin + form.targetPriceMax) / 2)) / 2500) * 1.1 +
          (form.hasAnc ? 2.4 : 0) +
          (form.hasFastCharge ? 2.8 : 0) +
          (form.isWireless ? 1.6 : 0)) /
        1.5
      ).toFixed(1)
    ),
    4,
    34
  )

  return {
    ...baseSchema,
    product_name: form.productName || baseSchema.product_name,
    category: form.category || baseSchema.category,
    discounted_price: Number(form.discountedPrice),
    actual_price: Number(form.actualPrice),
    discount_pct: discountPct,
    rating: Number((form.rating / 10).toFixed(2)),
    rating_count: ratingCount,
    market_channel: form.marketChannel || baseSchema.market_channel,
    is_wireless: form.isWireless ? 1 : 0,
    has_anc: form.hasAnc ? 1 : 0,
    has_enc: form.hasEnc ? 1 : 0,
    has_mic: form.hasMic ? 1 : 0,
    fast_charge: form.hasFastCharge ? 1 : 0,
    low_latency: form.hasLowLatency ? 1 : 0,
    voice_assistant: form.hasVoiceAssistant ? 1 : 0,
    bt_version: Number(form.bluetoothVersion || 5.3),
    ipx_rating: Number(form.ipxRating || 4),
    driver_mm: Number(form.driverMm || 12),
    playtime_hrs: Number(form.playtimeHrs || 24),
    sentiment_compound: Number(sentiment.toFixed(2)),
    pct_positive_reviews: Number(Math.min(0.98, (form.sentimentScore / 100) * 1.15).toFixed(2)),
    competitor_count: Number(form.competitorCount || 42),
    price_vs_median: Number(((form.discountedPrice / 1499) || 1).toFixed(2)),
    target_price_min: Number(form.targetPriceMin),
    target_price_max: Number(form.targetPriceMax),
    rating_velocity: Number(form.ratingVelocity),
    target_margin: Number(form.targetMargin || 22),
    demand_growth_projection: growth,
    opportunity_level: growth >= 18 ? 'High' : growth >= 11 ? 'Medium' : 'Low'
  }
}

export function calculateOpportunitySummary(schema) {
  const growth = Number(schema.demand_growth_projection || 0)
  const sentiment = Number(schema.sentiment_compound || 0) * 100
  const opportunityLevel = growth >= 18 ? 'High' : growth >= 11 ? 'Medium' : 'Low'

  return {
    level: opportunityLevel,
    growth,
    sentiment,
    signal: growth >= 18 ? 'Strong demand pull with healthy customer conversion.' : growth >= 11 ? 'Moderate upside with manageable competitive pressure.' : 'Demand remains soft; watch pricing and feature differentiation.'
  }
}

export const demoData = [
  { month: 'Feb', growth: 6.2 },
  { month: 'Mar', growth: 8.9 },
  { month: 'Apr', growth: 10.8 },
  { month: 'May', growth: 12.4 },
  { month: 'Jun', growth: 16.2 },
  { month: 'Jul', growth: 18.5 },
  { month: 'Aug', growth: 21.8 }
]

export const sentimentTrend = [
  { day: 'Mon', positive: 68, neutral: 18, negative: 14 },
  { day: 'Tue', positive: 71, neutral: 15, negative: 14 },
  { day: 'Wed', positive: 76, neutral: 12, negative: 12 },
  { day: 'Thu', positive: 82, neutral: 9, negative: 9 },
  { day: 'Fri', positive: 84, neutral: 8, negative: 8 },
  { day: 'Sat', positive: 86, neutral: 7, negative: 7 },
  { day: 'Sun', positive: 88, neutral: 5, negative: 7 }
]
