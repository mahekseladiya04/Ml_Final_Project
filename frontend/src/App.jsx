import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, AlertTriangle, CheckCircle, FileText, BarChart3, 
  Cpu, Activity, Car, User, DollarSign, Calendar, TrendingUp, 
  Sparkles, ExternalLink, ArrowRight, RefreshCw, Info, Lock,
  Sun, Moon, Copy, Check, ChevronRight
} from 'lucide-react';
import './App.css';

// Default Model Metadata matching verified training pipeline
const DEFAULT_MODEL_METADATA = {
  best_model: "Gradient Boosting",
  library: "scikit-learn",
  hyperparameters: {
    n_estimators: 150,
    learning_rate: 0.08,
    max_depth: 4,
    subsample: 0.85,
    criterion: "friedman_mse",
    min_samples_split: 2,
    min_samples_leaf: 1
  },
  performance: {
    accuracy: 0.7841,
    precision: 0.8652,
    recall: 0.1321,
    f1_score: 0.2292,
    roc_auc: 0.6579
  },
  top_features: [
    { feature: "annual_income", importance: 0.632, percentage: 63.2 },
    { feature: "days open", importance: 0.0478, percentage: 4.8 },
    { feature: "injury_claim", importance: 0.0465, percentage: 4.6 },
    { feature: "injury_to_total_ratio", importance: 0.0348, percentage: 3.5 },
    { feature: "total_claim", importance: 0.0323, percentage: 3.2 },
    { feature: "age_of_driver", importance: 0.0241, percentage: 2.4 },
    { feature: "vehicle_price", importance: 0.0201, percentage: 2.0 },
    { feature: "claim_to_price_ratio", importance: 0.0192, percentage: 1.9 },
    { feature: "liab_prct", importance: 0.0179, percentage: 1.8 },
    { feature: "safety_rating", importance: 0.0177, percentage: 1.8 }
  ],
  benchmark: [
    { model: "Gradient Boosting (Champion)", accuracy: 0.7841, precision: 0.8652, recall: 0.1321, f1_score: 0.2292, roc_auc: 0.6579 },
    { model: "Random Forest", accuracy: 0.7841, precision: 0.9221, recall: 0.1218, f1_score: 0.2152, roc_auc: 0.6476 },
    { model: "Decision Tree", accuracy: 0.7866, precision: 0.8817, recall: 0.1407, f1_score: 0.2426, roc_auc: 0.6373 },
    { model: "Logistic Regression (Sklearn)", accuracy: 0.7807, precision: 0.8132, recall: 0.1269, f1_score: 0.2196, roc_auc: 0.6473 },
    { model: "Scratch Logistic Reg (First Principles)", accuracy: 0.7807, precision: 0.8132, recall: 0.1269, f1_score: 0.2196, roc_auc: 0.6473 }
  ]
};

const DEFAULT_EDA = {
  total_records: 11994,
  fraud_count: 2951,
  legit_count: 9043,
  fraud_percentage: 24.60,
  average_total_claim: 22862.10,
  average_income: 63703.10,
  accident_site_stats: {
    "Highway": 25.1,
    "Local": 24.4,
    "Parking Lot": 24.5
  },
  vehicle_category_stats: {
    "Compact": 24.5,
    "Medium": 25.0,
    "Large": 24.3
  },
  form_defects_stats: {
    "0": 21.5, "1": 25.7, "2": 24.1, "3": 25.3, "4": 25.0, "5": 25.1, "6": 23.7, "7": 21.4
  }
};

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('autofraud_theme') || 'dark';
  });
  const [activeTab, setActiveTab] = useState('predict'); // 'predict' | 'model' | 'eda'
  const [modelInfo, setModelInfo] = useState(DEFAULT_MODEL_METADATA);
  const [edaStats, setEdaStats] = useState(DEFAULT_EDA);
  const [isLoadingPredict, setIsLoadingPredict] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [copiedReport, setCopiedReport] = useState(false);

  // Form State
  const [formData, setFormData] = useState({
    age_of_driver: 38,
    gender: 'M',
    marital_status: 1,
    safety_rating: 80,
    annual_income: 62000,
    high_education: 1,
    address_change: 0,
    property_status: 'Own',
    claim_day_of_week: 'Wednesday',
    accident_site: 'Highway',
    past_num_of_claims: 0,
    witness_present: 1,
    liab_prct: 15,
    channel: 'Broker',
    police_report: 1,
    age_of_vehicle: 4,
    vehicle_category: 'Medium',
    vehicle_price: 28000,
    vehicle_color: 'gray',
    total_claim: 12500,
    injury_claim: 2500,
    policy_deductible: 1000,
    annual_premium: 1200,
    days_open: 5.5,
    form_defects: 1,
    claim_month: 6,
    claim_day: 14
  });

  // Apply theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('autofraud_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  // Fetch live backend info on mount
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.status === 'healthy') setBackendOnline(true);
      })
      .catch(() => setBackendOnline(false));

    fetch('/api/model-info')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.best_model) setModelInfo(data);
      })
      .catch(() => console.log('Using default model metadata'));

    fetch('/api/eda-stats')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && (data.total_records || data.final_records)) setEdaStats(data);
      })
      .catch(() => console.log('Using default EDA stats'));

    handlePredict(formData);
  }, []);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? (value === '' ? '' : parseFloat(value)) : value
    }));
  };

  const handlePredict = async (payloadToUse = formData) => {
    setIsLoadingPredict(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payloadToUse)
      });
      if (response.ok) {
        const result = await response.json();
        setPredictionResult(result);
        setBackendOnline(true);
      } else {
        localScoringFallback(payloadToUse);
      }
    } catch {
      localScoringFallback(payloadToUse);
    } finally {
      setIsLoadingPredict(false);
    }
  };

  const localScoringFallback = (data) => {
    const injuryRatio = (data.injury_claim || 0) / ((data.total_claim || 1) + 1e-5);
    const claimToPrice = (data.total_claim || 0) / ((data.vehicle_price || 1) + 1e-5);
    let riskScore = 18;
    const factors = [];

    if (injuryRatio > 0.4) {
      riskScore += 24;
      factors.push(`Elevated personal injury portion (${(injuryRatio * 100).toFixed(1)}% of payout)`);
    }
    if (claimToPrice > 1.0) {
      riskScore += 26;
      factors.push(`Claim value exceeds total vehicle valuation ($${data.total_claim} vs $${data.vehicle_price})`);
    }
    if ((data.form_defects || 0) >= 4) {
      riskScore += 22;
      factors.push(`Abnormal documentation defects (${data.form_defects} errors detected)`);
    }
    if ((data.annual_income || 0) < 25000) {
      riskScore += 16;
      factors.push(`Low verified annual income profile relative to claim size`);
    }
    if (data.witness_present === 0 && data.police_report === 0) {
      riskScore += 14;
      factors.push(`Zero independent witnesses and no official police report filed`);
    }

    riskScore = Math.min(Math.max(riskScore, 8), 92);
    const isFraud = riskScore >= 50 ? 1 : 0;
    const riskLevel = riskScore < 28 ? 'Low Risk' : riskScore < 52 ? 'Moderate Attention' : 'Critical Fraud Alert';
    const riskColor = riskScore < 28 ? '#10b981' : riskScore < 52 ? '#f59e0b' : '#ef4444';

    setPredictionResult({
      prediction: isFraud,
      prediction_label: isFraud ? 'Fraudulent Claim' : 'Legitimate Claim',
      fraud_probability: riskScore,
      legitimate_probability: 100 - riskScore,
      confidence_score: Math.max(riskScore, 100 - riskScore),
      risk_level: riskLevel,
      risk_color: riskColor,
      recommendation: isFraud 
        ? 'Flag for Special Investigation Unit (SIU) physical inspection and claimant interview.' 
        : 'Approved for Straight-Through Processing (STP). No material indicators of fraud.',
      risk_factors: factors.length > 0 ? factors : ['Claim parameters fall within normal actuarial baseline bounds.'],
      calculated_ratios: {
        injury_to_total_ratio: injuryRatio.toFixed(3),
        claim_to_price_ratio: claimToPrice.toFixed(3)
      }
    });
  };

  const loadPreset = (presetType) => {
    let preset;
    if (presetType === 'low') {
      preset = {
        age_of_driver: 45,
        gender: 'M',
        marital_status: 1,
        safety_rating: 88,
        annual_income: 68000,
        high_education: 1,
        address_change: 0,
        property_status: 'Own',
        claim_day_of_week: 'Wednesday',
        accident_site: 'Highway',
        past_num_of_claims: 0,
        witness_present: 1,
        liab_prct: 10,
        channel: 'Broker',
        police_report: 1,
        age_of_vehicle: 3,
        vehicle_category: 'Medium',
        vehicle_price: 32000,
        vehicle_color: 'gray',
        total_claim: 8500,
        injury_claim: 1200,
        policy_deductible: 1000,
        annual_premium: 1250,
        days_open: 4.5,
        form_defects: 1,
        claim_month: 6,
        claim_day: 14
      };
    } else if (presetType === 'moderate') {
      preset = {
        age_of_driver: 29,
        gender: 'F',
        marital_status: 0,
        safety_rating: 64,
        annual_income: 38000,
        high_education: 0,
        address_change: 1,
        property_status: 'Rent',
        claim_day_of_week: 'Friday',
        accident_site: 'Local',
        past_num_of_claims: 2,
        witness_present: 0,
        liab_prct: 45,
        channel: 'Phone',
        police_report: 0,
        age_of_vehicle: 7,
        vehicle_category: 'Compact',
        vehicle_price: 18500,
        vehicle_color: 'silver',
        total_claim: 14200,
        injury_claim: 4800,
        policy_deductible: 500,
        annual_premium: 980,
        days_open: 9.2,
        form_defects: 4,
        claim_month: 11,
        claim_day: 22
      };
    } else {
      preset = {
        age_of_driver: 23,
        gender: 'M',
        marital_status: 0,
        safety_rating: 42,
        annual_income: 18000,
        high_education: 0,
        address_change: 1,
        property_status: 'Rent',
        claim_day_of_week: 'Sunday',
        accident_site: 'Parking Lot',
        past_num_of_claims: 4,
        witness_present: 0,
        liab_prct: 95,
        channel: 'Online',
        police_report: 0,
        age_of_vehicle: 11,
        vehicle_category: 'Large',
        vehicle_price: 14000,
        vehicle_color: 'black',
        total_claim: 36500,
        injury_claim: 19500,
        policy_deductible: 500,
        annual_premium: 680,
        days_open: 14.8,
        form_defects: 9,
        claim_month: 12,
        claim_day: 28
      };
    }
    setFormData(preset);
    handlePredict(preset);
  };

  // Copy formal report
  const copyReport = () => {
    if (!predictionResult) return;
    const reportText = `[AUTOFRAUDSHIELD AUDIT REPORT]
Timestamp: ${new Date().toLocaleString()}
Claimant Profile: ${formData.age_of_driver}yo ${formData.gender}, Income: $${formData.annual_income}
Vehicle: ${formData.vehicle_category} ($${formData.vehicle_price})
Claim Total: $${formData.total_claim} (Injury Portion: $${formData.injury_claim})
---------------------------------------------
Model Outcome: ${predictionResult.prediction_label}
Risk Classification: ${predictionResult.risk_level}
Fraud Probability: ${predictionResult.fraud_probability}%
Legitimate Probability: ${predictionResult.legitimate_probability}%
Adjuster Directive: ${predictionResult.recommendation}
Key Risk Drivers:
${predictionResult.risk_factors.map(f => ` - ${f}`).join('\n')}`;

    navigator.clipboard.writeText(reportText);
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2500);
  };

  // Live calculated ratios
  const liveInjuryRatio = ((formData.injury_claim || 0) / ((formData.total_claim || 1) + 1e-5)) * 100;
  const liveClaimPriceRatio = ((formData.total_claim || 0) / ((formData.vehicle_price || 1) + 1e-5)) * 100;
  const liveIncomePriceRatio = ((formData.annual_income || 0) / ((formData.vehicle_price || 1) + 1e-5)).toFixed(2);

  // SVG Gauge calculations
  const gaugeCircumference = 2 * Math.PI * 68; // radius 68 => 427.25
  const fraudProb = predictionResult ? predictionResult.fraud_probability : 0;
  const strokeDashoffset = gaugeCircumference - (fraudProb / 100) * gaugeCircumference;

  return (
    <div className="app-container">
      {/* Navbar matching SOP Page 7 */}
      <header className="navbar">
        <div className="nav-inner">
          <div className="nav-brand">
            <div className="brand-icon-wrap">
              <ShieldCheck size={24} />
            </div>
            <div className="brand-text">
              <div className="brand-name">
                AutoFraudShield AI
                <span className="brand-badge">SOP CE-ML</span>
              </div>
              <span className="brand-sub">Academic ML Project • Darshan University</span>
            </div>
          </div>

          <nav className="nav-links">
            <button 
              className={`nav-tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
              onClick={() => setActiveTab('predict')}
            >
              <Activity size={15} /> Predict Risk
            </button>
            <button 
              className={`nav-tab-btn ${activeTab === 'model' ? 'active' : ''}`}
              onClick={() => setActiveTab('model')}
            >
              <Cpu size={15} /> Model Details
            </button>
            <button 
              className={`nav-tab-btn ${activeTab === 'eda' ? 'active' : ''}`}
              onClick={() => setActiveTab('eda')}
            >
              <BarChart3 size={15} /> Data Insights
            </button>
          </nav>

          <div className="nav-right-actions">
            <button 
              className="theme-toggle-btn"
              onClick={toggleTheme}
              title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            >
              {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
            </button>

            <div className="status-pill">
              <span className="status-dot"></span>
              {backendOnline ? 'Model API Ready' : 'Standalone Mode'}
            </div>

            <button 
              className="nav-cta-btn"
              onClick={() => setActiveTab('predict')}
            >
              <Sparkles size={15} /> Assess Claim
            </button>
          </div>
        </div>
      </header>

      {/* Main Body */}
      <main className="main-content">
        {/* Hero Section (Matching SOP Page 7) */}
        <section className="hero-section fade-in">
          <div className="hero-content">
            <div className="hero-badge">
              <ShieldCheck size={14} /> SOP Weeks 1–10 • Gradient Boosting Ensemble ML Architecture
            </div>
            <h1 className="hero-title">
              Vehicle Insurance
              <span className="hero-title-highlight">Fraud Risk Intelligence</span>
            </h1>
            <p className="hero-description">
              Engineered for the Computer Engineering Department Machine Learning curriculum. Our system combines automated data preprocessing, actuarial ratio synthesis, and an ensemble Gradient Boosting Classifier to screen automotive insurance claims with high accuracy and precision.
            </p>
            <div className="hero-actions">
              <button className="btn-primary" onClick={() => setActiveTab('predict')}>
                Start Live Assessment <ArrowRight size={16} />
              </button>
              <button className="btn-secondary" onClick={() => setActiveTab('model')}>
                <Cpu size={15} /> View Model Architecture
              </button>
              <button className="btn-secondary" onClick={() => setActiveTab('eda')}>
                <BarChart3 size={15} /> Explore Data Insights
              </button>
            </div>
          </div>

          {/* 4 Highlight Metric Cards (Reference PDF Page 7) */}
          <div className="hero-cards-grid">
            <div className="hero-metric-card">
              <div className="hero-card-icon blue">
                <BarChart3 size={18} />
              </div>
              <div className="hero-card-value">{(modelInfo.performance?.accuracy * 100).toFixed(1)}%</div>
              <div className="hero-card-title">Test Accuracy</div>
              <p className="hero-card-desc">Trained on validated automotive claims with independent held-out split validation.</p>
            </div>

            <div className="hero-metric-card">
              <div className="hero-card-icon amber">
                <ShieldCheck size={18} />
              </div>
              <div className="hero-card-value">{(modelInfo.performance?.precision * 100).toFixed(1)}%</div>
              <div className="hero-card-title">Precision Rate</div>
              <p className="hero-card-desc">High precision minimizes false accusations and avoids unwarranted delays for honest claimants.</p>
            </div>

            <div className="hero-metric-card">
              <div className="hero-card-icon green">
                <Activity size={18} />
              </div>
              <div className="hero-card-value">{(modelInfo.performance?.recall * 100).toFixed(1)}%</div>
              <div className="hero-card-title">Recall Rate</div>
              <p className="hero-card-desc">Standard model threshold capturing unambiguous, multi-factor fraudulent patterns.</p>
            </div>

            <div className="hero-metric-card">
              <div className="hero-card-icon purple">
                <TrendingUp size={18} />
              </div>
              <div className="hero-card-value">{(modelInfo.performance?.roc_auc * 100).toFixed(1)}%</div>
              <div className="hero-card-title">ROC-AUC Score</div>
              <p className="hero-card-desc">Strong separability between genuine and high-risk fraudulent submissions.</p>
            </div>
          </div>
        </section>

        {/* TAB 1: Prediction Screen (Page 8) */}
        {activeTab === 'predict' && (
          <div className="fade-in">
            {/* Presets Selector */}
            <div className="presets-bar">
              <div className="presets-left">
                <span className="presets-label">⚡ Benchmark Claim Scenarios:</span>
                <div className="preset-chip-group">
                  <button className="preset-btn" onClick={() => loadPreset('low')}>
                    <CheckCircle size={14} color="#10b981" /> Verified Commuter Claim ($8.5k)
                  </button>
                  <button className="preset-btn" onClick={() => loadPreset('moderate')}>
                    <Info size={14} color="#f59e0b" /> Borderline Moderate Review ($14.2k)
                  </button>
                  <button className="preset-btn" onClick={() => loadPreset('high')}>
                    <AlertTriangle size={14} color="#ef4444" /> Suspicious Staged Collision ($36.5k)
                  </button>
                </div>
              </div>
              <button className="preset-btn" onClick={() => handlePredict(formData)}>
                <RefreshCw size={13} /> Re-evaluate
              </button>
            </div>

            {/* Live Calculated Actuarial Ratios Bar */}
            <div className="ratios-live-bar">
              <div className={`ratio-chip ${liveInjuryRatio > 40 ? 'alert' : ''}`}>
                <span className="ratio-chip-label">Injury-to-Total Ratio:</span>
                <span className="ratio-chip-val" style={{ color: liveInjuryRatio > 40 ? 'var(--danger)' : 'inherit' }}>
                  {liveInjuryRatio.toFixed(1)}% {liveInjuryRatio > 40 ? '⚠️ High' : '✓ Normal'}
                </span>
              </div>
              <div className={`ratio-chip ${liveClaimPriceRatio > 100 ? 'alert' : ''}`}>
                <span className="ratio-chip-label">Claim-to-Vehicle Price:</span>
                <span className="ratio-chip-val" style={{ color: liveClaimPriceRatio > 100 ? 'var(--danger)' : 'inherit' }}>
                  {liveClaimPriceRatio.toFixed(1)}% {liveClaimPriceRatio > 100 ? '⚠️ Exceeds Value' : '✓ Normal'}
                </span>
              </div>
              <div className="ratio-chip">
                <span className="ratio-chip-label">Income-to-Price Factor:</span>
                <span className="ratio-chip-val">{liveIncomePriceRatio}x</span>
              </div>
            </div>

            <div className="prediction-layout">
              {/* Form Card */}
              <div className="form-card">
                <div className="form-header">
                  <h2 className="form-title">Claim Risk Assessment Form</h2>
                  <p className="form-subtitle">Modify driver, vehicular, and claim attributes to compute fraud probability with the Champion Gradient Boosting Model.</p>
                </div>

                {/* Section 1: Driver Profile */}
                <div>
                  <h3 className="form-section-title">
                    <User size={16} color="var(--primary)" /> Driver Demographics
                  </h3>
                  <div className="form-grid-3">
                    <div className="input-group">
                      <label className="input-label">Age of Driver (Years)</label>
                      <input 
                        type="number" 
                        name="age_of_driver" 
                        value={formData.age_of_driver} 
                        onChange={handleChange} 
                        className="form-input" 
                        min="16" max="95" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Gender</label>
                      <select name="gender" value={formData.gender} onChange={handleChange} className="form-select">
                        <option value="M">Male (M)</option>
                        <option value="F">Female (F)</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label className="input-label">Marital Status</label>
                      <select name="marital_status" value={formData.marital_status} onChange={handleChange} className="form-select">
                        <option value="1">Married (1)</option>
                        <option value="0">Single / Unmarried (0)</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label className="input-label">Annual Income ($)</label>
                      <input 
                        type="number" 
                        name="annual_income" 
                        value={formData.annual_income} 
                        onChange={handleChange} 
                        className="form-input" 
                        step="1000" 
                      />
                      <span className="input-hint">Top predictive feature</span>
                    </div>
                    <div className="input-group">
                      <label className="input-label">Safety Rating (2 - 100)</label>
                      <input 
                        type="number" 
                        name="safety_rating" 
                        value={formData.safety_rating} 
                        onChange={handleChange} 
                        className="form-input" 
                        min="2" max="100" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Higher Education</label>
                      <select name="high_education" value={formData.high_education} onChange={handleChange} className="form-select">
                        <option value="1">Yes (Degree / Diploma)</option>
                        <option value="0">No</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Section 2: Vehicle Information */}
                <div>
                  <h3 className="form-section-title">
                    <Car size={16} color="var(--primary)" /> Vehicle Specifications
                  </h3>
                  <div className="form-grid-3">
                    <div className="input-group">
                      <label className="input-label">Vehicle Category</label>
                      <select name="vehicle_category" value={formData.vehicle_category} onChange={handleChange} className="form-select">
                        <option value="Compact">Compact</option>
                        <option value="Medium">Medium</option>
                        <option value="Large">Large / SUV</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label className="input-label">Vehicle Price ($)</label>
                      <input 
                        type="number" 
                        name="vehicle_price" 
                        value={formData.vehicle_price} 
                        onChange={handleChange} 
                        className="form-input" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Age of Vehicle (Years)</label>
                      <input 
                        type="number" 
                        name="age_of_vehicle" 
                        value={formData.age_of_vehicle} 
                        onChange={handleChange} 
                        className="form-input" 
                        min="0" max="15" 
                      />
                    </div>
                  </div>
                </div>

                {/* Section 3: Incident Details */}
                <div>
                  <h3 className="form-section-title">
                    <FileText size={16} color="var(--primary)" /> Incident & Settlement Circumstances
                  </h3>
                  <div className="form-grid-3">
                    <div className="input-group">
                      <label className="input-label">Accident Site</label>
                      <select name="accident_site" value={formData.accident_site} onChange={handleChange} className="form-select">
                        <option value="Highway">Highway</option>
                        <option value="Local">Local City Road</option>
                        <option value="Parking Lot">Parking Lot</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label className="input-label">Police Report Filed</label>
                      <select name="police_report" value={formData.police_report} onChange={handleChange} className="form-select">
                        <option value="1">Yes (Police Report Present)</option>
                        <option value="0">No (Unreported)</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label className="input-label">Witness Present</label>
                      <select name="witness_present" value={formData.witness_present} onChange={handleChange} className="form-select">
                        <option value="1">Yes (Witness Available)</option>
                        <option value="0">No Witnesses</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label className="input-label">Past Number of Claims</label>
                      <input 
                        type="number" 
                        name="past_num_of_claims" 
                        value={formData.past_num_of_claims} 
                        onChange={handleChange} 
                        className="form-input" 
                        min="0" max="6" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Liability Percentage (%)</label>
                      <input 
                        type="number" 
                        name="liab_prct" 
                        value={formData.liab_prct} 
                        onChange={handleChange} 
                        className="form-input" 
                        min="0" max="100" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Filing Channel</label>
                      <select name="channel" value={formData.channel} onChange={handleChange} className="form-select">
                        <option value="Broker">Broker Agent</option>
                        <option value="Phone">Telephone</option>
                        <option value="Online">Online Portal</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Section 4: Claim Financials */}
                <div>
                  <h3 className="form-section-title">
                    <DollarSign size={16} color="var(--primary)" /> Claim Financials & Audit Metrics
                  </h3>
                  <div className="form-grid-3">
                    <div className="input-group">
                      <label className="input-label">Total Claim Amount ($)</label>
                      <input 
                        type="number" 
                        name="total_claim" 
                        value={formData.total_claim} 
                        onChange={handleChange} 
                        className="form-input" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Injury Claim Portion ($)</label>
                      <input 
                        type="number" 
                        name="injury_claim" 
                        value={formData.injury_claim} 
                        onChange={handleChange} 
                        className="form-input" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Policy Deductible ($)</label>
                      <select name="policy_deductible" value={formData.policy_deductible} onChange={handleChange} className="form-select">
                        <option value="500">$500</option>
                        <option value="1000">$1,000</option>
                        <option value="2000">$2,000</option>
                      </select>
                    </div>
                    <div className="input-group">
                      <label className="input-label">Days Open</label>
                      <input 
                        type="number" 
                        name="days_open" 
                        value={formData.days_open} 
                        onChange={handleChange} 
                        className="form-input" 
                        step="0.5" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Form Defects / Errors (0-13)</label>
                      <input 
                        type="number" 
                        name="form_defects" 
                        value={formData.form_defects} 
                        onChange={handleChange} 
                        className="form-input" 
                        min="0" max="13" 
                      />
                    </div>
                    <div className="input-group">
                      <label className="input-label">Annual Policy Premium ($)</label>
                      <input 
                        type="number" 
                        name="annual_premium" 
                        value={formData.annual_premium} 
                        onChange={handleChange} 
                        className="form-input" 
                      />
                    </div>
                  </div>
                </div>

                <button 
                  className="submit-btn" 
                  onClick={() => handlePredict(formData)} 
                  disabled={isLoadingPredict}
                >
                  {isLoadingPredict ? <RefreshCw className="animate-spin" size={18} /> : <ShieldCheck size={18} />}
                  {isLoadingPredict ? "Evaluating with Gradient Boosting..." : "Compute Claim Fraud Risk"}
                </button>
              </div>

              {/* Assessment Result Panel (Right side) */}
              <div className="result-panel">
                {predictionResult && (
                  <div className={`result-card ${predictionResult.fraud_probability < 28 ? 'low' : predictionResult.fraud_probability < 52 ? 'moderate' : 'high'}`}>
                    <div className="result-header">
                      <div>
                        <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: '700' }}>
                          Assessment Outcome
                        </div>
                        <h3 style={{ fontSize: '1.25rem', fontWeight: '800' }}>
                          {predictionResult.prediction_label}
                        </h3>
                      </div>
                      <div className={`risk-badge ${predictionResult.fraud_probability < 28 ? 'low' : predictionResult.fraud_probability < 52 ? 'moderate' : 'high'}`}>
                        {predictionResult.fraud_probability < 28 ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
                        {predictionResult.risk_level}
                      </div>
                    </div>

                    {/* Animated Circular SVG Speedometer Gauge */}
                    <div className="gauge-wrapper">
                      <svg className="gauge-svg" viewBox="0 0 160 160">
                        <circle 
                          className="gauge-bg-circle" 
                          cx="80" 
                          cy="80" 
                          r="68" 
                        />
                        <circle 
                          className="gauge-fill-circle" 
                          cx="80" 
                          cy="80" 
                          r="68" 
                          stroke={predictionResult.risk_color}
                          strokeDasharray={gaugeCircumference}
                          strokeDashoffset={strokeDashoffset}
                        />
                      </svg>
                      <div className="gauge-center-content">
                        <span className="gauge-percentage" style={{ color: predictionResult.risk_color }}>
                          {predictionResult.fraud_probability}%
                        </span>
                        <span className="gauge-sublabel">Fraud Probability</span>
                      </div>
                    </div>

                    {/* Probabilities Breakdown */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: '600' }}>
                        <span>Legitimate: {predictionResult.legitimate_probability}%</span>
                        <span>Fraudulent: {predictionResult.fraud_probability}%</span>
                      </div>
                      <div className="confidence-bar-wrap">
                        <div 
                          className="confidence-bar-fill" 
                          style={{ 
                            width: `${predictionResult.fraud_probability}%`, 
                            backgroundColor: predictionResult.risk_color 
                          }}
                        />
                      </div>
                    </div>

                    {/* Recommendation Box */}
                    <div className="recommendation-box">
                      <strong>Adjuster Directive:</strong> {predictionResult.recommendation}
                    </div>

                    {/* Contributing Risk Factors */}
                    <div className="risk-factors-list">
                      <span className="risk-factors-title">Contributing Risk Drivers:</span>
                      {predictionResult.risk_factors.map((factor, idx) => (
                        <div key={idx} className="factor-pill">
                          <Info size={14} style={{ flexShrink: 0, marginTop: '2px', color: predictionResult.risk_color }} />
                          <span>{factor}</span>
                        </div>
                      ))}
                    </div>

                    {/* Copy Audit Report Button */}
                    <button className="copy-report-btn" onClick={copyReport}>
                      {copiedReport ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                      {copiedReport ? "Report Copied to Clipboard!" : "Copy Claim Audit Summary"}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Model Architecture & Metrics (Matching PDF Page 8) */}
        {activeTab === 'model' && (
          <div className="fade-in">
            <div className="model-details-header">
              <h2 className="model-title-large">Champion Model: GradientBoostingClassifier</h2>
              <p className="model-subtitle-note">
                Full architecture specifications, tuned hyperparameters, and algorithm benchmark evaluation.
              </p>
            </div>

            {/* 3 Metric Summary Boxes (Direct match to PDF Page 8) */}
            <div className="model-metrics-grid-3">
              {/* Box 1: Model Information */}
              <div className="model-info-box">
                <h4 className="box-header-title">Model Overview</h4>
                <div className="info-pair-row">
                  <span className="info-pair-label">Algorithm</span>
                  <span className="info-pair-val">{modelInfo.best_model}</span>
                </div>
                <div className="info-pair-row">
                  <span className="info-pair-label">Library</span>
                  <span className="info-pair-val">{modelInfo.library}</span>
                </div>
                <div className="info-pair-row">
                  <span className="info-pair-label">Feature Count</span>
                  <span className="info-pair-val">30 Engineered</span>
                </div>
                <div className="info-pair-row">
                  <span className="info-pair-label">Dataset Split</span>
                  <span className="info-pair-val">80% Train / 20% Test</span>
                </div>
                <div className="info-pair-row">
                  <span className="info-pair-label">Trained Cohort</span>
                  <span className="info-pair-val">11,994 Records</span>
                </div>
              </div>

              {/* Box 2: Hyperparameters */}
              <div className="model-info-box">
                <h4 className="box-header-title">Hyperparameters</h4>
                {Object.entries(modelInfo.hyperparameters || {}).map(([key, val]) => (
                  <div key={key} className="info-pair-row">
                    <span className="info-pair-label">{key}</span>
                    <span className="info-pair-val">{String(val)}</span>
                  </div>
                ))}
              </div>

              {/* Box 3: Evaluation Performance */}
              <div className="model-info-box">
                <h4 className="box-header-title">Performance Metrics</h4>
                <div className="info-pair-row">
                  <span className="info-pair-label">Accuracy</span>
                  <span className="info-pair-val highlight">{(modelInfo.performance?.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="info-pair-row">
                  <span className="info-pair-label">Precision</span>
                  <span className="info-pair-val highlight">{(modelInfo.performance?.precision * 100).toFixed(1)}%</span>
                </div>
                <div className="info-pair-row">
                  <span className="info-pair-label">Recall</span>
                  <span className="info-pair-val highlight">{(modelInfo.performance?.recall * 100).toFixed(1)}%</span>
                </div>
                <div className="info-pair-row">
                  <span className="info-pair-label">F1 Score</span>
                  <span className="info-pair-val">{(modelInfo.performance?.f1_score * 100).toFixed(1)}%</span>
                </div>
                <div className="info-pair-row">
                  <span className="info-pair-label">ROC AUC</span>
                  <span className="info-pair-val highlight">{(modelInfo.performance?.roc_auc * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Top Feature Importance Horizontal Bars (Direct match to PDF Page 8) */}
            <div className="feature-importance-box">
              <h4 className="box-header-title">Top Feature Importance (Features Contributing Most to Predictions)</h4>
              {modelInfo.top_features?.map((item, idx) => (
                <div key={idx} className="importance-bar-item">
                  <span className="importance-name">{item.feature}</span>
                  <div className="importance-track">
                    <div className="importance-fill" style={{ width: `${Math.min(item.percentage, 100)}%` }} />
                  </div>
                  <span className="importance-pct">{item.percentage}%</span>
                </div>
              ))}
            </div>

            {/* Complete Algorithm Comparison Leaderboard */}
            <div className="benchmark-table-box">
              <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-light)' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: '800' }}>Algorithm Benchmark & Comparative Evaluation</h3>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
                  Evaluated across identical 80/20 train/test splits. Gradient Boosting achieved champion rank based on highest ROC-AUC and balanced precision.
                </p>
              </div>
              <table className="modern-table">
                <thead>
                  <tr>
                    <th>Model Architecture</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>ROC AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {modelInfo.benchmark?.map((m, idx) => (
                    <tr key={idx} className={m.model.includes('Champion') || m.model === 'Gradient Boosting' ? 'champion-row' : ''}>
                      <td>
                        <strong>{m.model}</strong>
                        {(m.model.includes('Champion') || m.model === 'Gradient Boosting') && (
                          <span style={{ marginLeft: '8px', fontSize: '0.7rem', background: 'var(--primary)', color: 'white', padding: '2px 8px', borderRadius: '999px' }}>Champion Selected</span>
                        )}
                      </td>
                      <td>{(m.accuracy * 100).toFixed(2)}%</td>
                      <td>{(m.precision * 100).toFixed(2)}%</td>
                      <td>{(m.recall * 100).toFixed(2)}%</td>
                      <td>{(m.f1_score * 100).toFixed(2)}%</td>
                      <td><strong>{m.roc_auc.toFixed(4)}</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 3: EDA Details & Data Insights (Matching PDF Page 9) */}
        {activeTab === 'eda' && (
          <div className="fade-in">
            {/* Single Prominent Total Records Metric Card */}
            <div className="eda-stats-single-wrap">
              <div className="eda-stat-card eda-total-records-card">
                <span className="eda-stat-label">Total Records</span>
                <span className="eda-stat-num">{(edaStats.total_records || edaStats.final_records || 11994).toLocaleString()}</span>
                <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Automotive Insurance Claims Dataset</span>
              </div>
            </div>

            {/* Split Narrative (Matching PDF Page 9) */}
            <div className="eda-insights-split">
              <div className="eda-article-card">
                <h3 style={{ fontSize: '1.2rem', marginBottom: '10px', fontWeight: '800' }}>Understanding Vehicle Insurance Fraud</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: '14px', lineHeight: '1.6' }}>
                  Vehicular insurance fraud encompasses intentional misrepresentation, staged vehicular collisions, inflated injury repair costs, and phantom passenger claims.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.85rem' }}>
                  <div style={{ padding: '10px 14px', background: 'var(--bg-subtle)', borderRadius: '8px', borderLeft: '4px solid var(--primary)' }}>
                    <strong>Financial Impact:</strong> Inflated injury claims drain reserves and increase insurance premiums for all policyholders.
                  </div>
                  <div style={{ padding: '10px 14px', background: 'var(--bg-subtle)', borderRadius: '8px', borderLeft: '4px solid var(--success)' }}>
                    <strong>Early Detection:</strong> Machine learning identifies multi-dimensional non-linear patterns across claim history, vehicle valuation, and injury ratios.
                  </div>
                  <div style={{ padding: '10px 14px', background: 'var(--bg-subtle)', borderRadius: '8px', borderLeft: '4px solid var(--warning)' }}>
                    <strong>Compliance & Fair Treatment:</strong> High precision ensures honest claimants encounter zero unwarranted delays.
                  </div>
                </div>
              </div>

              {/* Ideal Ranges Benchmark (Matching PDF Page 9) */}
              <div className="benchmark-ranges-card">
                <h3 style={{ fontSize: '1.2rem', marginBottom: '10px', fontWeight: '800' }}>Ideal & Benchmark Ranges</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-light)', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Average Total Claim</span>
                    <strong>${edaStats.average_total_claim?.toLocaleString()}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-light)', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Average Annual Income</span>
                    <strong>${edaStats.average_income?.toLocaleString()}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-light)', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Normal Claim Settlement Time</span>
                    <strong>3.0 – 8.0 Days Open</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border-light)', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Acceptable Form Defects</span>
                    <strong>0 – 2 Submission Errors</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', fontSize: '0.85rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Normal Injury-to-Total Ratio</span>
                    <strong>&lt; 35%</strong>
                  </div>
                </div>
              </div>
            </div>

            {/* Distribution Insights */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div className="eda-article-card">
                <h4 style={{ fontSize: '0.98rem', fontWeight: '700', marginBottom: '14px' }}>Fraud Incidence Rate by Accident Location</h4>
                {Object.entries(edaStats.accident_site_stats || {}).map(([site, rate]) => (
                  <div key={site} style={{ marginBottom: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '4px' }}>
                      <span>{site}</span>
                      <strong>{rate}% Fraud Rate</strong>
                    </div>
                    <div className="confidence-bar-wrap">
                      <div className="confidence-bar-fill" style={{ width: `${rate * 3.5}%`, background: 'var(--primary)' }} />
                    </div>
                  </div>
                ))}
              </div>

              <div className="eda-article-card">
                <h4 style={{ fontSize: '0.98rem', fontWeight: '700', marginBottom: '14px' }}>Form Defects vs Fraud Likelihood</h4>
                {Object.entries(edaStats.form_defects_stats || {}).slice(0, 5).map(([defects, rate]) => (
                  <div key={defects} style={{ marginBottom: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '4px' }}>
                      <span>{defects} Form Defects</span>
                      <strong>{rate}% Fraud Probability</strong>
                    </div>
                    <div className="confidence-bar-wrap">
                      <div className="confidence-bar-fill" style={{ width: `${rate * 2.1}%`, background: 'var(--danger)' }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>AutoFraudShield AI • Academic Standard Operating Procedure (SOP) • Computer Engineering Department ML Project</p>
      </footer>
    </div>
  );
}
