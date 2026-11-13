document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const uploadForm = document.getElementById("uploadForm");
  const btnText = document.getElementById("btnText");
  const btnSpinner = document.getElementById("btnSpinner");
  const errorAlert = document.getElementById("errorAlert");

  // Results elements
  const resultsSection = document.getElementById("resultsSection");
  const predictionBadge = document.getElementById("predictionBadge");
  const caseIdDisplay = document.getElementById("caseIdDisplay");
  const confidenceValue = document.getElementById("confidenceValue");
  const origImgDisplay = document.getElementById("origImgDisplay");
  const gradcamImgDisplay = document.getElementById("gradcamImgDisplay");
  const radarImgDisplay = document.getElementById("radarImgDisplay");
  const hgnnImgDisplay = document.getElementById("hgnnImgDisplay");
  const conceptsContainer = document.getElementById("conceptsContainer");
  const vltNarrativeDisplay = document.getElementById("vltNarrativeDisplay");
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");

  let selectedFile = null;

  if (dropzone && fileInput) {
    // Dropzone click
    dropzone.addEventListener("click", () => fileInput.click());

    // Drag & Drop
    ["dragenter", "dragover"].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileSelect(files[0]);
      }
    });

    // File input change
    fileInput.addEventListener("change", (e) => {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelect(fileInput.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    
    if (!validTypes.includes(file.type) && ![".jpg", ".jpeg", ".png", ".webp"].includes(ext)) {
      showError("Invalid format. Please upload a JPG, JPEG, PNG, or WebP retinal fundus image.");
      return;
    }

    selectedFile = file;
    hideError();

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewContainer.style.display = "block";
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Quick sample loader
  window.loadSample = function(sampleType) {
    fetch(`/api/sample/${sampleType}`)
      .then(res => {
        if (!res.ok) throw new Error("Sample unavailable");
        return res.blob();
      })
      .then(blob => {
        const file = new File([blob], `sample_${sampleType}.png`, { type: "image/png" });
        handleFileSelect(file);
      })
      .catch(err => {
        showError("Could not load sample: " + err.message);
      });
  };

  // Form submission
  if (uploadForm) {
    uploadForm.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!selectedFile) {
        showError("Please select an image first.");
        return;
      }

      setLoading(true);
      hideError();

      const formData = new FormData();
      formData.append("image", selectedFile);
      const patientId = document.getElementById("patientId")?.value || "ANON-PATIENT";
      formData.append("patient_id", patientId);

      fetch("/predict", {
        method: "POST",
        body: formData
      })
      .then(async (res) => {
        const contentType = res.headers.get("content-type") || "";
        let data;
        if (contentType.includes("application/json")) {
          data = await res.json();
        } else {
          const text = await res.text();
          if (res.status === 502 || res.status === 504) {
            throw new Error("The cloud server was spinning up from idle (cold start). Please click 'Run Deep Multimodal Screening' again now that the server is active.");
          }
          throw new Error(`Server returned HTTP ${res.status}. Please retry your submission.`);
        }
        if (!res.ok) {
          throw new Error(data.detail || data.error || data.message || `Error ${res.status}: Analysis failed.`);
        }
        return data;
      })
      .then(data => {
        setLoading(false);
        renderResults(data);
      })
      .catch(err => {
        setLoading(false);
        showError(err.message || "An unexpected error occurred during inference.");
      });
    });
  }

  function renderResults(data) {
    if (!resultsSection) return;

    // 1. Prediction Badge & Confidence
    const isGlaucoma = data.prediction && data.prediction.toLowerCase().includes("glaucoma");
    if (predictionBadge) {
      if (isGlaucoma) {
        predictionBadge.innerHTML = `<span class="prediction-badge-glaucoma">● GLAUCOMA DETECTED</span>`;
      } else {
        predictionBadge.innerHTML = `<span class="prediction-badge-normal">● NORMAL / NON-GLAUCOMATOUS</span>`;
      }
    }

    if (caseIdDisplay) {
      caseIdDisplay.textContent = data.case_id;
    }

    if (confidenceValue) {
      const pct = (data.confidence * 100).toFixed(1);
      confidenceValue.textContent = `${pct}%`;
      confidenceValue.style.color = isGlaucoma ? "var(--rose-main)" : "var(--emerald-main)";
    }

    // 2. Inspection Images (preprocessed, Grad-CAM, Radar)
    if (origImgDisplay && data.orig_url) {
      origImgDisplay.src = data.orig_url;
    }
    if (gradcamImgDisplay && data.gradcam_url) {
      gradcamImgDisplay.src = data.gradcam_url;
    }
    if (radarImgDisplay && data.radar_url) {
      radarImgDisplay.src = data.radar_url;
    }

    // 3. X-HGNN Sub-hypergraph Grounding
    if (hgnnImgDisplay && data.hgnn_url) {
      hgnnImgDisplay.src = data.hgnn_url;
    }

    // 4. Clinical Concepts Grid
    if (conceptsContainer && data.clinical_concepts) {
      conceptsContainer.innerHTML = "";
      data.clinical_concepts.forEach(concept => {
        const isAlert = concept.badge === "danger";
        const barColor = isAlert ? "var(--rose-main)" : "var(--emerald-main)";
        const statusColor = isAlert ? "var(--rose-main)" : "var(--emerald-main)";

        const itemEl = document.createElement("div");
        itemEl.className = "concept-item";
        itemEl.innerHTML = `
          <div class="concept-item-header">
            <span class="concept-title">${escapeHtml(concept.title)}</span>
            <span class="concept-score">${concept.score_pct}%</span>
          </div>
          <div class="concept-bar-bg">
            <div class="concept-bar-fill" style="width: ${Math.min(100, Math.max(0, concept.score_pct))}%; background: ${barColor};"></div>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
            <span style="font-size: 0.72rem; font-weight: 700; color: ${statusColor}; text-transform: uppercase;">
              ${escapeHtml(concept.status)}
            </span>
            <span style="font-size: 0.7rem; color: var(--text-dim); font-family: monospace;">score: ${concept.score}</span>
          </div>
          <div class="concept-desc">${escapeHtml(concept.description)}</div>
        `;
        conceptsContainer.appendChild(itemEl);
      });
    }

    // 5. Vision-Language Transformer Narrative
    if (vltNarrativeDisplay) {
      vltNarrativeDisplay.textContent = data.vlt_explanation || "No narrative generated.";
    }

    // 6. PDF summary link
    if (downloadPdfBtn && data.pdf_url) {
      downloadPdfBtn.href = data.pdf_url;
    }

    // 7. Reveal and smooth scroll
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function setLoading(loading) {
    if (loading) {
      analyzeBtn.disabled = true;
      btnSpinner.style.display = "inline-block";
      btnText.textContent = "Analyzing Fundus Structure...";
    } else {
      analyzeBtn.disabled = false;
      btnSpinner.style.display = "none";
      btnText.textContent = "Analyze Fundus Image";
    }
  }

  function showError(msg) {
    if (errorAlert) {
      errorAlert.textContent = msg;
      errorAlert.style.display = "block";
    } else {
      alert(msg);
    }
  }

  function hideError() {
    if (errorAlert) {
      errorAlert.style.display = "none";
    }
  }

  function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
});
