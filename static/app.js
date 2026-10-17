/**
 * app.js - Production Frontend Controller for Market Opportunity Analyzer
 * Connects with FastAPI backend:
 * - /api/analysis/{id} (Executive Overview)
 * - /api/simulate (Interactive What-If Scenario Simulator)
 * - /api/model/feature-importance (XGBoost Gain & Weight ranking)
 * - /api/dataset/products (Master Competitor Benchmarks)
 * - /api/export/report/{id} (Executive Dossier generation)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Chart Instances
  let demandChart = null;
  let sentimentChart = null;
  let priceScatterChart = null;
  let searchTrendChart = null;
  let productRadarChart = null;
  let featureImportanceChart = null;

  // Global State
  let currentProductData = null;
  let masterCompetitors = [];

  // DOM Elements - Navigation & Tabs
  const navItems = document.querySelectorAll(".sidebar-menu .nav-item");
  const tabViews = document.querySelectorAll(".tab-view");
  const searchInput = document.getElementById("productSearchInput");
  const searchDropdown = document.getElementById("searchDropdown");
  const datePickerBtn = document.getElementById("datePickerBtn");
  const dateMenu = document.getElementById("dateMenu");
  const currentDateRange = document.getElementById("currentDateRange");
  const exportDossierBtn = document.getElementById("exportDossierBtn");

  // DOM Elements - Top 4 Summary Cards
  const cardMarketOpportunity = document.getElementById("cardMarketOpportunity");
  const cardMarketOpportunitySub = document.getElementById("cardMarketOpportunitySub");
  const cardDemandGrowth = document.getElementById("cardDemandGrowth");
  const cardDemandGrowthSub = document.getElementById("cardDemandGrowthSub");
  const cardSentimentScore = document.getElementById("cardSentimentScore");
  const cardSentimentScoreSub = document.getElementById("cardSentimentScoreSub");
  const cardPriceRange = document.getElementById("cardPriceRange");
  const cardPriceRangeSub = document.getElementById("cardPriceRangeSub");

  // DOM Elements - Hero Product Card
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

  // DOM Elements - Charts Badges & Sentiment
  const demandGrowthBadge = document.getElementById("demandGrowthBadge");
  const searchTrendBadge = document.getElementById("searchTrendBadge");
  const donutReviewsCount = document.getElementById("donutReviewsCount");
  const posRatio = document.getElementById("posRatio");
  const neuRatio = document.getElementById("neuRatio");
  const negRatio = document.getElementById("negRatio");
  const aspectsList = document.getElementById("aspectsList");
  const competitionTableBody = document.getElementById("competitionTableBody");

  // DOM Elements - Simulator
  const simTargetPrice = document.getElementById("simTargetPrice");
  const valTargetPrice = document.getElementById("valTargetPrice");
  const simActualPrice = document.getElementById("simActualPrice");
  const valActualPrice = document.getElementById("valActualPrice");
  const simDiscountDisplay = document.getElementById("simDiscountDisplay");
  const simBattery = document.getElementById("simBattery");
  const valBattery = document.getElementById("valBattery");
  const simRating = document.getElementById("simRating");
  const valRating = document.getElementById("valRating");
  const simAnc = document.getElementById("simAnc");
  const simEnc = document.getElementById("simEnc");
  const simFastCharge = document.getElementById("simFastCharge");
  const simLowLatency = document.getElementById("simLowLatency");
  const runSimulationBtn = document.getElementById("runSimulationBtn");

  const simPredGrowthDisplay = document.getElementById("simPredGrowthDisplay");
  const simTierDisplay = document.getElementById("simTierDisplay");
  const simScoreDisplay = document.getElementById("simScoreDisplay");
  const simCorridorDisplay = document.getElementById("simCorridorDisplay");
  const simElasticityDisplay = document.getElementById("simElasticityDisplay");
  const simStrengthsList = document.getElementById("simStrengthsList");
  const simRisksList = document.getElementById("simRisksList");

  // DOM Elements - Master Competition & Export Modal
  const masterCompTableBody = document.getElementById("masterCompTableBody");
  const compTableFilter = document.getElementById("compTableFilter");
  const exportModal = document.getElementById("exportModal");
  const closeExportModal = document.getElementById("closeExportModal");
  const dismissModalBtn = document.getElementById("dismissModalBtn");
  const printReportBtn = document.getElementById("printReportBtn");
  const dossierTitle = document.getElementById("dossierTitle");
  const dossierVerdict = document.getElementById("dossierVerdict");
  const dossierScore = document.getElementById("dossierScore");
  const dossierGrowth = document.getElementById("dossierGrowth");
  const dossierCorridor = document.getElementById("dossierCorridor");
  const dossierSentiment = document.getElementById("dossierSentiment");

  // =================================================================
  // 1. Tab Navigation Routing
  // =================================================================
  navItems.forEach(item => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const tabId = item.getAttribute("data-tab");

      navItems.forEach(n => n.classList.remove("active"));
      item.classList.add("active");

      tabViews.forEach(view => {
        if (view.id === `view-${tabId}`) {
          view.classList.remove("hidden");
          view.classList.add("active");
        } else {
          view.classList.add("hidden");
          view.classList.remove("active");
        }
      });

      // Lazy load chart views
      if (tabId === "product-analysis") {
        initProductRadarChart();
      } else if (tabId === "model-monitoring") {
        loadFeatureImportance();
      } else if (tabId === "price-competition") {
        loadMasterCompetitionData();
      }
    });
  });

  // Date Range Dropdown Toggle
  datePickerBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    dateMenu.classList.toggle("hidden");
  });

  document.querySelectorAll(".dropdown-opt").forEach(opt => {
    opt.addEventListener("click", () => {
      document.querySelectorAll(".dropdown-opt").forEach(o => o.classList.remove("active"));
      opt.classList.add("active");
      currentDateRange.textContent = opt.getAttribute("data-range");
      dateMenu.classList.add("hidden");
    });
  });

  document.addEventListener("click", () => {
    dateMenu.classList.add("hidden");
  });

  // =================================================================
  // 2. Chart.js Dashboard Visualizations
  // =================================================================

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

  // =================================================================
  // 3. Tab 2: Radar Competency Chart
  // =================================================================
  function initProductRadarChart() {
    const canvas = document.getElementById("productRadarChart");
    if (!canvas) return;
    if (productRadarChart) return; // already initialized

    const ctx = canvas.getContext("2d");
    productRadarChart = new Chart(ctx, {
      type: "radar",
      data: {
        labels: ["Acoustic Bass", "Battery Life", "Mic Call Clarity", "Ergonomics", "Low Latency", "Value for Money"],
        datasets: [
          {
            label: "Boat Airdopes 141",
            data: [84, 88, 62, 75, 78, 92],
            fill: true,
            backgroundColor: "rgba(37, 99, 235, 0.2)",
            borderColor: "#2563eb",
            pointBackgroundColor: "#2563eb",
            pointBorderColor: "#fff",
            borderWidth: 2
          },
          {
            label: "Category Benchmark",
            data: [70, 68, 65, 70, 60, 72],
            fill: true,
            backgroundColor: "rgba(148, 163, 184, 0.15)",
            borderColor: "#94a3b8",
            pointBackgroundColor: "#94a3b8",
            borderDash: [4, 4],
            borderWidth: 1.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: "#e2e8f0" },
            grid: { color: "#f1f5f9" },
            suggestedMin: 30,
            suggestedMax: 100,
            ticks: { display: false }
          }
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: { font: { family: "Inter", size: 11 } }
          }
        }
      }
    });
  }

  // =================================================================
  // 4. Tab 8: Feature Importance Horizontal Bar Chart
  // =================================================================
  async function loadFeatureImportance() {
    const canvas = document.getElementById("featureImportanceChart");
    if (!canvas) return;
    if (featureImportanceChart) return;

    try {
      const res = await fetch("/api/model/feature-importance");
      if (!res.ok) return;
      const data = await res.json();
      const top = data.top_features || [];

      const labels = top.map(f => f.name);
      const values = top.map(f => f.importance_pct);
      const colors = top.map(f => {
        if (f.category.includes("Pricing")) return "#2563eb";
        if (f.category.includes("Sentiment")) return "#10b981";
        if (f.category.includes("Hardware")) return "#8b5cf6";
        return "#f59e0b";
      });

      const ctx = canvas.getContext("2d");
      featureImportanceChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "XGBoost Gain Contribution (%)",
              data: values,
              backgroundColor: colors,
              borderRadius: 4
            }
          ]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => ` Importance: ${ctx.raw}%`
              }
            }
          },
          scales: {
            x: {
              grid: { color: "#f1f5f9" },
              ticks: { callback: v => `${v}%`, font: { family: "Inter", size: 11 }, color: "#64748b" }
            },
            y: {
              grid: { display: false },
              ticks: { font: { family: "Inter", size: 11, weight: 500 }, color: "#1e293b" }
            }
          }
        }
      });
    } catch (e) {
      console.error("Failed to load feature importance:", e);
    }
  }

  // =================================================================
  // 5. Tab 5: Master Competition Table Loader
  // =================================================================
  async function loadMasterCompetitionData() {
    try {
      const res = await fetch("/api/dataset/products?limit=15");
      if (!res.ok) return;
      const data = await res.json();
      masterCompetitors = data.items || [];
      renderMasterCompetitionTable(masterCompetitors);
    } catch (e) {
      console.error("Error fetching master products:", e);
    }
  }

  function renderMasterCompetitionTable(items) {
    if (!masterCompTableBody) return;
    masterCompTableBody.innerHTML = "";
    items.forEach(item => {
      const discount = Math.round(((item.actual_price - item.price) / item.actual_price) * 100);
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${item.brand}</strong></td>
        <td>${item.model}</td>
        <td><strong>₹${item.price.toLocaleString()}</strong></td>
        <td><span style="color:#94a3b8; text-decoration:line-through;">₹${item.actual_price.toLocaleString()}</span></td>
        <td><span class="badge badge-green">${discount}% OFF</span></td>
        <td>${item.rating} ★</td>
        <td>${item.reviews.toLocaleString()}</td>
        <td>${item.anc ? '<span class="value-green">Yes</span>' : '<span style="color:#94a3b8">No</span>'}</td>
        <td>${item.battery}h</td>
        <td><button class="table-action-btn" data-brand="${item.brand.toLowerCase()}">Analyze</button></td>
      `;

      tr.querySelector(".table-action-btn").addEventListener("click", () => {
        loadProductAnalysis(item.brand.toLowerCase());
        // Switch to dashboard tab
        document.querySelector('[data-tab="dashboard"]').click();
      });

      masterCompTableBody.appendChild(tr);
    });
  }

  if (compTableFilter) {
    compTableFilter.addEventListener("input", (e) => {
      const term = e.target.value.toLowerCase().trim();
      const filtered = masterCompetitors.filter(p => 
        p.brand.toLowerCase().includes(term) || p.model.toLowerCase().includes(term)
      );
      renderMasterCompetitionTable(filtered);
    });
  }

  // =================================================================
  // 6. Interactive What-If Scenario Simulator
  // =================================================================
  function updateImpliedDiscount() {
    const tp = parseFloat(simTargetPrice.value);
    const mrp = parseFloat(simActualPrice.value);
    if (mrp > tp) {
      const disc = Math.round(((mrp - tp) / mrp) * 100);
      simDiscountDisplay.textContent = `Implied Consumer Discount: ${disc}%`;
    } else {
      simDiscountDisplay.textContent = "Price is at or above list MRP";
    }
  }

  simTargetPrice.addEventListener("input", (e) => {
    valTargetPrice.textContent = `₹${parseInt(e.target.value).toLocaleString()}`;
    updateImpliedDiscount();
  });

  simActualPrice.addEventListener("input", (e) => {
    valActualPrice.textContent = `₹${parseInt(e.target.value).toLocaleString()}`;
    updateImpliedDiscount();
  });

  simBattery.addEventListener("input", (e) => {
    valBattery.textContent = `${e.target.value} hrs`;
  });

  simRating.addEventListener("input", (e) => {
    valRating.textContent = `${parseFloat(e.target.value).toFixed(1)} ★`;
  });

  runSimulationBtn.addEventListener("click", async () => {
    const payload = {
      product_name: currentProductData ? currentProductData.product_info.full_name : "Simulated Product",
      target_price: parseFloat(simTargetPrice.value),
      actual_price: parseFloat(simActualPrice.value),
      has_anc: simAnc.checked,
      has_enc: simEnc.checked,
      fast_charge: simFastCharge.checked,
      low_latency: simLowLatency.checked,
      battery_hrs: parseFloat(simBattery.value),
      rating_expected: parseFloat(simRating.value)
    };

    runSimulationBtn.innerHTML = `<span>Computing XGBoost Inference...</span>`;

    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        const sim = await res.json();
        simPredGrowthDisplay.textContent = sim.predicted_demand_growth;
        simTierDisplay.textContent = sim.market_opportunity_tier;
        simTierDisplay.className = `highlight-val-large ${sim.market_opportunity_tier === 'HIGH' ? 'value-green' : 'value-blue'}`;
        simScoreDisplay.textContent = `Score: ${sim.opportunity_score} / 100`;
        simCorridorDisplay.textContent = sim.optimal_price_corridor.formatted;
        simElasticityDisplay.textContent = sim.price_elasticity;

        simStrengthsList.innerHTML = sim.strengths.map(s => `<li>✔ ${s}</li>`).join("");
        simRisksList.innerHTML = sim.risks.map(r => `<li>⚠ ${r}</li>`).join("");
      }
    } catch (err) {
      console.error("Simulation request failed:", err);
    } finally {
      runSimulationBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        <span>Run Real-time XGBoost Prediction</span>
      `;
    }
  });

  // =================================================================
  // 7. Update Circular Gauge
  // =================================================================
  function updateGauge(score) {
    gaugeScoreNumber.textContent = score;
    // Circumference = 2 * PI * 48 = 301.6
    const circumference = 301.6;
    const offset = circumference - (score / 100) * circumference;
    gaugeProgress.style.strokeDashoffset = offset;
  }

  // =================================================================
  // 8. Render Main Dashboard Data
  // =================================================================
  function renderDashboard(data) {
    currentProductData = data;
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

    // Badges & Sentiment
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

    // Sync Simulator sliders with current product
    simTargetPrice.value = p.current_price;
    valTargetPrice.textContent = `₹${p.current_price.toLocaleString()}`;
    simActualPrice.value = p.actual_price;
    valActualPrice.textContent = `₹${p.actual_price.toLocaleString()}`;
    updateImpliedDiscount();
  }

  // =================================================================
  // 9. API Fetch & Autocomplete Search
  // =================================================================
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

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".nav-search-container")) {
      searchDropdown.classList.add("hidden");
    }
  });

  // =================================================================
  // 10. Executive Dossier Modal & Export
  // =================================================================
  exportDossierBtn.addEventListener("click", async () => {
    const prodKey = currentProductData ? currentProductData.product_info.id : "boat";
    try {
      const res = await fetch(`/api/export/report/${encodeURIComponent(prodKey)}`);
      if (res.ok) {
        const rep = await res.json();
        dossierTitle.textContent = rep.title;
        dossierVerdict.textContent = rep.executive_summary.verdict;
        dossierScore.textContent = `${rep.executive_summary.score} / 100`;
        dossierGrowth.textContent = rep.executive_summary.predicted_demand_growth;
        dossierCorridor.textContent = rep.executive_summary.optimal_pricing_corridor;
        dossierSentiment.textContent = `${rep.executive_summary.sentiment_score} Positive`;
        exportModal.classList.remove("hidden");
      }
    } catch (e) {
      console.error("Export report fetch failed:", e);
    }
  });

  closeExportModal.addEventListener("click", () => exportModal.classList.add("hidden"));
  dismissModalBtn.addEventListener("click", () => exportModal.classList.add("hidden"));
  printReportBtn.addEventListener("click", () => window.print());

  // Initial Load
  loadProductAnalysis("default");
});
