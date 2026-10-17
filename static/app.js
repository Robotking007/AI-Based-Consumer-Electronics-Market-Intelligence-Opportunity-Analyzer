/**
 * app.js - Frontend Controller for Market Opportunity Analyzer Dashboard
 * Connects with FastAPI backend (/api/analysis, /api/products, /api/predict)
 * Renders high-fidelity Chart.js visualizations matching target design.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Chart instances
  let demandChart = null;
  let sentimentChart = null;
  let priceScatterChart = null;
  let searchTrendChart = null;

  // DOM Elements
  const searchInput = document.getElementById("productSearchInput");
  const searchDropdown = document.getElementById("searchDropdown");
  const competitionTableBody = document.getElementById("competitionTableBody");

  // Top metric elements
  const cardMarketOpportunity = document.getElementById("cardMarketOpportunity");
  const cardMarketOpportunitySub = document.getElementById("cardMarketOpportunitySub");
  const cardDemandGrowth = document.getElementById("cardDemandGrowth");
  const cardDemandGrowthSub = document.getElementById("cardDemandGrowthSub");
  const cardSentimentScore = document.getElementById("cardSentimentScore");
  const cardSentimentScoreSub = document.getElementById("cardSentimentScoreSub");
  const cardPriceRange = document.getElementById("cardPriceRange");
  const cardPriceRangeSub = document.getElementById("cardPriceRangeSub");

  // Hero card elements
  const heroProductTitle = document.getElementById("heroProductTitle");
  const heroProductImg = document.getElementById("heroProductImg");
  const specBrand = document.getElementById("specBrand");
  const specModel = document.getElementById("specModel");
  const specCategory = document.getElementById("specCategory");
  const specCurrentPrice = document.getElementById("specCurrentPrice");
  const specDiscount = document.getElementById("specDiscount");
  const specRating = document.getElementById("specRating");
  const specTotalReviews = document.getElementById("specTotalReviews");
  const recHeadline = document.getElementById("recHeadline");
  const gaugeScoreNumber = document.getElementById("gaugeScoreNumber");
  const gaugeProgress = document.getElementById("gaugeProgress");
  const gaugeStatusLabel = document.getElementById("gaugeStatusLabel");

  // Chart badge elements
  const demandGrowthBadge = document.getElementById("demandGrowthBadge");
  const searchTrendBadge = document.getElementById("searchTrendBadge");
  const donutReviewsCount = document.getElementById("donutReviewsCount");
  const posRatio = document.getElementById("posRatio");
  const neuRatio = document.getElementById("neuRatio");
  const negRatio = document.getElementById("negRatio");
  const aspectsList = document.getElementById("aspectsList");

  // Sidebar navigation
  const navItems = document.querySelectorAll(".nav-item");
  const modelMonitoringSection = document.getElementById("modelMonitoringSection");

  navItems.forEach(item => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      navItems.forEach(n => n.classList.remove("active"));
      item.classList.add("active");

      const tab = item.getAttribute("data-tab");
      if (tab === "model-monitoring") {
        if (modelMonitoringSection) {
          modelMonitoringSection.style.display = "block";
          modelMonitoringSection.scrollIntoView({ behavior: "smooth" });
        }
      } else {
        if (modelMonitoringSection) {
          modelMonitoringSection.style.display = "none";
        }
      }
    });
  });

  // -----------------------------------------------------------------
  // 1. Chart.js Initialization
  // -----------------------------------------------------------------

  function initDemandGrowthChart(labels, predictedData, historicalData) {
    const ctx = document.getElementById("demandGrowthChart").getContext("2d");
    if (demandChart) demandChart.destroy();

    demandChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Predicted Demand",
            data: predictedData,
            borderColor: "#2563eb",
            backgroundColor: "#2563eb",
            borderWidth: 2.2,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.35,
            fill: false
          },
          {
            label: "Historical Trend",
            data: historicalData,
            borderColor: "#94a3b8",
            borderWidth: 1.8,
            borderDash: [4, 4],
            pointRadius: 0,
            pointHoverRadius: 0,
            tension: 0.35,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0f172a",
            titleFont: { family: "Inter", size: 12 },
            bodyFont: { family: "Inter", size: 12 },
            padding: 10,
            cornerRadius: 6
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: "Inter", size: 11 }, color: "#64748b" }
          },
          y: {
            min: 0,
            max: 200,
            ticks: {
              stepSize: 50,
              font: { family: "Inter", size: 11 },
              color: "#64748b"
            },
            grid: { color: "#f1f5f9" },
            title: {
              display: true,
              text: "Demand Index",
              font: { family: "Inter", size: 11, weight: 500 },
              color: "#64748b"
            }
          }
        }
      }
    });
  }

  function initSentimentDonutChart(positive, neutral, negative) {
    const ctx = document.getElementById("sentimentDonutChart").getContext("2d");
    if (sentimentChart) sentimentChart.destroy();

    sentimentChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Positive", "Neutral", "Negative"],
        datasets: [
          {
            data: [positive, neutral, negative],
            backgroundColor: ["#10b981", "#cbd5e1", "#f87171"],
            borderWidth: 0,
            hoverOffset: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "74%",
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => ` ${context.label}: ${context.raw}%`
            }
          }
        }
      }
    });
  }

  function initPricePositionScatter(selectedProduct, competitors) {
    const ctx = document.getElementById("pricePositionChart").getContext("2d");
    if (priceScatterChart) priceScatterChart.destroy();

    const selectedData = [{ x: selectedProduct.price, y: selectedProduct.rating, name: selectedProduct.name }];
    const compData = competitors.slice(0, 7).map(c => ({ x: c.price, y: c.rating, name: c.name }));
    const similarData = competitors.slice(7).map(c => ({ x: c.price, y: c.rating, name: c.name }));

    priceScatterChart = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Selected Product",
            data: selectedData,
            backgroundColor: "#ef4444",
            borderColor: "#fee2e2",
            borderWidth: 4,
            pointRadius: 7,
            pointHoverRadius: 9
          },
          {
            label: "Competition",
            data: compData,
            backgroundColor: "#3b82f6",
            pointRadius: 5,
            pointHoverRadius: 7
          },
          {
            label: "Similar Products",
            data: similarData,
            backgroundColor: "#93c5fd",
            pointRadius: 4,
            pointHoverRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const pt = context.raw;
                return ` ${pt.name || 'Product'}: ₹${pt.x.toLocaleString()} (${pt.y}★)`;
              }
            }
          }
        },
        scales: {
          x: {
            min: 0,
            max: 5000,
            ticks: {
              stepSize: 1000,
              callback: (v) => v === 0 ? "0" : `${v.toLocaleString()}`,
              font: { family: "Inter", size: 10 },
              color: "#64748b"
            },
            grid: { color: "#f1f5f9" },
            title: {
              display: true,
              text: "Price (₹)",
              font: { family: "Inter", size: 10 },
              color: "#64748b"
            }
          },
          y: {
            min: 0,
            max: 5,
            ticks: {
              stepSize: 1,
              font: { family: "Inter", size: 10 },
              color: "#64748b"
            },
            grid: { color: "#f1f5f9" },
            title: {
              display: true,
              text: "Rating",
              font: { family: "Inter", size: 10 },
              color: "#64748b"
            }
          }
        }
      }
    });
  }

  function initSearchTrendChart(labels, values) {
    const ctx = document.getElementById("searchTrendChart").getContext("2d");
    if (searchTrendChart) searchTrendChart.destroy();

    searchTrendChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Search Interest",
            data: values,
            borderColor: "#8b5cf6",
            backgroundColor: "#8b5cf6",
            borderWidth: 2,
            pointRadius: 3.5,
            pointHoverRadius: 5.5,
            tension: 0.35,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: "Inter", size: 10 }, color: "#64748b" }
          },
          y: {
            min: 0,
            max: 100,
            ticks: {
              stepSize: 25,
              font: { family: "Inter", size: 10 },
              color: "#64748b"
            },
            grid: { color: "#f1f5f9" },
            title: {
              display: true,
              text: "Search Interest",
              font: { family: "Inter", size: 10 },
              color: "#64748b"
            }
          }
        }
      }
    });
  }

  // -----------------------------------------------------------------
  // 2. Update Gauge Score
  // -----------------------------------------------------------------
  function updateGauge(score) {
    gaugeScoreNumber.textContent = score;
    // Circumference = 2 * PI * 48 = 301.6
    const circumference = 301.6;
    const offset = circumference - (score / 100) * circumference;
    gaugeProgress.style.strokeDashoffset = offset;
  }

  // -----------------------------------------------------------------
  // 3. Render Dashboard Data from Backend
  // -----------------------------------------------------------------
  function renderDashboard(data) {
    const p = data.product_info;
    const sm = data.summary_metrics;
    const opp = data.opportunity_score;

    // Top metrics
    cardMarketOpportunity.textContent = sm.market_opportunity;
    cardMarketOpportunitySub.textContent = sm.market_opportunity_sub;
    cardDemandGrowth.textContent = sm.predicted_demand_growth;
    cardDemandGrowthSub.textContent = sm.predicted_demand_growth_sub;
    cardSentimentScore.textContent = sm.sentiment_score;
    cardSentimentScoreSub.textContent = sm.sentiment_score_sub;
    cardPriceRange.textContent = sm.optimal_price_range;
    cardPriceRangeSub.textContent = sm.optimal_price_range_sub;

    // Hero Product
    heroProductTitle.textContent = `Product Analysis – ${p.full_name}`;
    if (p.image_url) heroProductImg.src = p.image_url;
    specBrand.textContent = p.brand;
    specModel.textContent = p.model;
    specCategory.textContent = p.category;
    specCurrentPrice.textContent = `₹${p.current_price.toLocaleString()}`;
    specDiscount.textContent = `${p.discount_percent}% (was ₹${p.actual_price.toLocaleString()})`;
    specRating.textContent = `${p.rating} / 5`;
    specTotalReviews.textContent = p.total_reviews.toLocaleString();

    recHeadline.textContent = opp.recommendation;
    gaugeStatusLabel.textContent = opp.label;
    updateGauge(opp.score);

    // Badges
    demandGrowthBadge.textContent = data.demand_growth_chart.badge;
    searchTrendBadge.textContent = data.search_trends.badge;
    donutReviewsCount.textContent = data.sentiment_analysis.total_reviews;
    posRatio.textContent = `${data.sentiment_analysis.breakdown.positive}%`;
    neuRatio.textContent = `${data.sentiment_analysis.breakdown.neutral}%`;
    negRatio.textContent = `${data.sentiment_analysis.breakdown.negative}%`;

    // Aspects
    aspectsList.innerHTML = "";
    data.sentiment_analysis.aspects.forEach(asp => {
      const pct = Math.round(asp.score * 100);
      const row = document.createElement("div");
      row.className = "aspect-row";
      row.innerHTML = `
        <span class="aspect-name">${asp.aspect}</span>
        <div class="aspect-bar-track">
          <div class="aspect-bar-fill" style="width: ${pct}%; background-color: ${asp.color};"></div>
        </div>
        <span class="aspect-score">${asp.score.toFixed(2)}</span>
      `;
      aspectsList.appendChild(row);
    });

    // Competition table
    competitionTableBody.innerHTML = "";
    data.competition_analysis.forEach(item => {
      const tr = document.createElement("tr");
      if (item.is_selected) tr.className = "row-selected";
      tr.innerHTML = `
        <td>${item.brand}</td>
        <td>${item.model}</td>
        <td>${item.price.toLocaleString()}</td>
        <td>${item.rating}</td>
        <td>${item.reviews}</td>
      `;
      tr.addEventListener("click", () => {
        loadProductAnalysis(item.brand.toLowerCase());
      });
      competitionTableBody.appendChild(tr);
    });

    // Charts
    initDemandGrowthChart(
      data.demand_growth_chart.labels,
      data.demand_growth_chart.predicted,
      data.demand_growth_chart.historical
    );

    initSentimentDonutChart(
      data.sentiment_analysis.breakdown.positive,
      data.sentiment_analysis.breakdown.neutral,
      data.sentiment_analysis.breakdown.negative
    );

    initPricePositionScatter(
      data.price_positioning.selected_product,
      data.price_positioning.competitors
    );

    initSearchTrendChart(
      data.search_trends.labels,
      data.search_trends.values
    );
  }

  // -----------------------------------------------------------------
  // 4. API Fetch Functions
  // -----------------------------------------------------------------
  async function loadProductAnalysis(productId = "default") {
    try {
      const res = await fetch(`/api/analysis/${encodeURIComponent(productId)}`);
      if (res.ok) {
        const data = await res.json();
        renderDashboard(data);
      }
    } catch (e) {
      console.error("Failed to fetch product analysis:", e);
    }
  }

  // Search input handler
  let debounceTimer;
  searchInput.addEventListener("input", (e) => {
    clearTimeout(debounceTimer);
    const query = e.target.value.trim();
    if (!query) {
      searchDropdown.classList.add("hidden");
      return;
    }

    debounceTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/products?q=${encodeURIComponent(query)}`);
        if (res.ok) {
          const items = await res.json();
          if (items.length > 0) {
            searchDropdown.innerHTML = items.map(p => `
              <div class="dropdown-item" data-id="${p.id || p.name}">
                <span>${p.name}</span>
                <span class="item-price">₹${p.price ? p.price.toLocaleString() : ''}</span>
              </div>
            `).join("");
            searchDropdown.classList.remove("hidden");

            // Attach click
            searchDropdown.querySelectorAll(".dropdown-item").forEach(item => {
              item.addEventListener("click", () => {
                const id = item.getAttribute("data-id");
                searchInput.value = item.querySelector("span").textContent;
                searchDropdown.classList.add("hidden");
                loadProductAnalysis(id);
              });
            });
          } else {
            searchDropdown.classList.add("hidden");
          }
        }
      } catch (err) {
        console.error("Error searching products:", err);
      }
    }, 250);
  });

  // Close dropdown on outside click
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".nav-search-container")) {
      searchDropdown.classList.add("hidden");
    }
  });

  // Initial Load
  loadProductAnalysis("default");
});
