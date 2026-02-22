document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const loading = document.getElementById('loading');
    const resultsContent = document.getElementById('results-content');

    // Tabs and Camera
    const tabUpload = document.getElementById('tab-upload');
    const tabCamera = document.getElementById('tab-camera');
    const cameraArea = document.getElementById('camera-area');
    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');

    // Theme Toggle
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = themeToggle.querySelector('i');

    // Stats Elements
    const totalCountEl = document.getElementById('total-count');
    const pointsEl = document.getElementById('user-points');
    const levelEl = document.getElementById('user-level');
    const lastScanEl = document.getElementById('last-scan');

    // Result Elements
    const resLabel = document.getElementById('res-label');
    const resConf = document.getElementById('res-conf');
    const resTime = document.getElementById('res-time');
    const resDesc = document.getElementById('res-desc');

    const bars = {
        'plastik': { fill: document.getElementById('fill-p'), text: document.getElementById('prob-p') },
        'kertas': { fill: document.getElementById('fill-k'), text: document.getElementById('prob-k') },
        'kaca': { fill: document.getElementById('fill-g'), text: document.getElementById('prob-g') },
        'logam': { fill: document.getElementById('fill-m'), text: document.getElementById('prob-m') }
    };

    const explanations = {
        "kaca": "Kaca merupakan bahan anorganik yang dapat didaur ulang tanpa mengurangi kualitasnya. Material ini tidak terurai secara alami, namun 100% dapat didaur ulang berulang kali.",
        "kertas": "Kertas berasal dari bahan selulosa. Daur ulang kertas sangat penting karena dapat menghemat ribuan liter air dan menyelamatkan banyak pohon. Waktu terurainya sekitar 2-6 minggu.",
        "logam": "Logam seperti kaleng aluminium dapat didaur ulang dan digunakan kembali dalam industri. Membutuhkan waktu sekitar 50 tahun untuk terurai di alam.",
        "plastik": "Plastik sangat sulit terurai (100-1000 tahun) dan dapat mencemari ekosistem laut. Sangat disarankan untuk mendaur ulang melalui bank sampah atau sistem pengumpulan resmi."
    };

    let stream = null;
    let statsChart = null;

    const tagLocationBtn = document.getElementById('tag-location');

    // --- Location Logic ---
    tagLocationBtn.addEventListener('click', () => {
        if ("geolocation" in navigator) {
            tagLocationBtn.disabled = true;
            tagLocationBtn.textContent = "Mencari Lokasi...";
            navigator.geolocation.getCurrentPosition((position) => {
                const { latitude, longitude } = position.coords;
                let stats = JSON.parse(localStorage.getItem('green_cycle_stats') || '{"total":0, "points":0, "history":[]}');
                if (stats.history.length > 0) {
                    stats.history[stats.history.length - 1].location = { lat: latitude, lng: longitude };
                    localStorage.setItem('green_cycle_stats', JSON.stringify(stats));
                    alert(`Lokasi berhasil ditandai: ${latitude.toFixed(4)}, ${longitude.toFixed(4)}`);
                }
                tagLocationBtn.textContent = "Lokasi Berhasil Ditandai";
            }, (err) => {
                alert("Gagal mengambil lokasi: " + err.message);
                tagLocationBtn.disabled = false;
                tagLocationBtn.textContent = "Tandai Lokasi";
            });
        }
    });

    // --- initialization ---
    initTheme();
    initStats();

    // --- Theme Logic ---
    function initTheme() {
        if (localStorage.getItem('theme') === 'dark') {
            document.body.classList.add('dark-mode');
            themeIcon.className = 'fas fa-sun';
        }
    }

    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        themeIcon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        if (statsChart) updateChartColors();
    });

    // --- Stats & Gamification Logic ---
    function initStats() {
        const stats = JSON.parse(localStorage.getItem('green_cycle_stats') || '{"total":0, "points":0, "history":[]}');
        const trashTypes = JSON.parse(localStorage.getItem('trash_types') || '{"plastik":0, "kertas":0, "kaca":0, "logam":0}');

        totalCountEl.textContent = stats.total;
        pointsEl.textContent = stats.points;
        levelEl.textContent = getLevelName(stats.points);
        if (stats.history.length > 0) {
            lastScanEl.textContent = stats.history[stats.history.length - 1].time;
        }

        renderChart(trashTypes);
    }

    function getLevelName(points) {
        if (points >= 500) return "Eco Guardian";
        if (points >= 200) return "Eco Warrior";
        if (points >= 50) return "Eco Friend";
        return "Pemula";
    }

    function updateStats(label) {
        let stats = JSON.parse(localStorage.getItem('green_cycle_stats') || '{"total":0, "points":0, "history":[]}');
        let trashTypes = JSON.parse(localStorage.getItem('trash_types') || '{"plastik":0, "kertas":0, "kaca":0, "logam":0}');

        if (label !== 'manusia' && label !== 'unknown') {
            stats.total += 1;
            stats.points += 10;
            const now = new Date();
            stats.history.push({ label, time: now.toLocaleString() });
            if (stats.history.length > 10) stats.history.shift(); // Keep last 10

            if (trashTypes.hasOwnProperty(label)) {
                trashTypes[label] += 1;
            }

            localStorage.setItem('green_cycle_stats', JSON.stringify(stats));
            localStorage.setItem('trash_types', JSON.stringify(trashTypes));

            // UI Update
            totalCountEl.textContent = stats.total;
            pointsEl.textContent = stats.points;
            levelEl.textContent = getLevelName(stats.points);
            lastScanEl.textContent = stats.history[stats.history.length - 1].time;

            updateChart(trashTypes);
        }
    }

    function renderChart(data) {
        const ctx = document.getElementById('statsChart').getContext('2d');
        const theme = document.body.classList.contains('dark-mode') ? 'dark' : 'light';
        const colors = theme === 'dark' ? '#e0e0e0' : '#333333';

        statsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(data).map(k => k.charAt(0).toUpperCase() + k.slice(1)),
                datasets: [{
                    label: 'Jumlah Terdeteksi',
                    data: Object.values(data),
                    backgroundColor: ['#4caf50', '#81c784', '#aed581', '#2e7d32'],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: colors } },
                    x: { ticks: { color: colors } }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function updateChart(data) {
        statsChart.data.datasets[0].data = Object.values(data);
        statsChart.update();
    }

    function updateChartColors() {
        const theme = document.body.classList.contains('dark-mode') ? 'dark' : 'light';
        const colors = theme === 'dark' ? '#e0e0e0' : '#333333';
        statsChart.options.scales.y.ticks.color = colors;
        statsChart.options.scales.x.ticks.color = colors;
        statsChart.update();
    }

    // --- Tab & Camera Logic ---
    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('active');
        tabCamera.classList.remove('active');
        dropZone.classList.remove('hidden');
        cameraArea.classList.add('hidden');
        stopCamera();
    });

    tabCamera.addEventListener('click', () => {
        tabCamera.classList.add('active');
        tabUpload.classList.remove('active');
        dropZone.classList.add('hidden');
        cameraArea.classList.remove('hidden');
        // previewContainer.classList.add('hidden'); // Removed so results stay visible
        startCamera();
    });

    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            webcamElement.srcObject = stream;
        } catch (err) {
            console.error("Camera access denied: ", err);
            alert("Tidak dapat mengakses kamera. Pastikan izin kamera sudah diberikan.");
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    captureBtn.addEventListener('click', () => {
        const context = canvasElement.getContext('2d');
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
        context.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);
        canvasElement.toBlob((blob) => {
            const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
            handleCapturedFile(file, canvasElement.toDataURL('image/jpeg'));
        }, 'image/jpeg');
    });

    function handleCapturedFile(file, dataUrl) {
        imagePreview.src = dataUrl;
        cameraArea.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        loading.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        stopCamera();
        classifyImage(file);
    }

    // --- Upload Logic ---
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, () => dropZone.classList.remove('drag-over'));
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    resetBtn.addEventListener('click', () => {
        previewContainer.classList.add('hidden');
        if (tabUpload.classList.contains('active')) {
            dropZone.classList.remove('hidden');
        } else {
            cameraArea.classList.remove('hidden');
            startCamera();
        }
        tagLocationBtn.disabled = false;
        tagLocationBtn.innerHTML = '<i class="fas fa-map-marker-alt"></i> Tandai Lokasi';
        fileInput.value = '';
    });

    async function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Harap unggah file gambar yang valid.');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            loading.classList.remove('hidden');
            resultsContent.classList.add('hidden');
            classifyImage(file);
        };
        reader.readAsDataURL(file);
    }

    async function classifyImage(file) {
        loading.classList.remove('hidden');
        resultsContent.classList.add('hidden');

        try {
            // --- NEW: Client-side resizing to avoid Vercel 4.5MB limit ---
            const resizedBlob = await new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = 224; // Model input size
                    canvas.height = 224;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, 224, 224);
                    canvas.toBlob(resolve, 'image/jpeg', 0.82);
                };
                img.src = URL.createObjectURL(file);
            });

            const formData = new FormData();
            formData.append('file', resizedBlob, 'input.jpg');

            const response = await fetch('/api/classify', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Server Error ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);
            updateStats(data.label);
        } catch (error) {
            console.error(error);
            alert(`Terjadi kesalahan: ${error.message}. Pastikan server backend berjalan.`);
            loading.classList.add('hidden');
        }
    }

    function displayResults(data) {
        loading.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        if (data.status === "error_server") {
            resLabel.textContent = "Error Server";
            resDesc.textContent = `Error: ${data.message}`;
            return;
        }

        if (data.status === "error_human") {
            resLabel.textContent = "Status: Manusia";
            resConf.textContent = "-";
            resTime.textContent = data.time + 's';
            resDesc.textContent = "Terdeteksi wajah manusia. Sistem hanya mengklasifikasi sampah.";
            resetBars();
            return;
        }

        if (data.status === "uncertain") {
            resLabel.textContent = "Tidak Terkelompokkan";
            resDesc.textContent = "Sistem tidak yakin dengan jenis sampah ini.";
        } else {
            resLabel.textContent = data.label;
            resConf.textContent = (data.confidence * 100).toFixed(1) + '%';
            resTime.textContent = data.time + 's';
            resDesc.textContent = explanations[data.label] || "";
        }

        Object.keys(data.probabilities).forEach(key => {
            const prob = data.probabilities[key];
            const percent = (prob * 100).toFixed(0) + '%';
            if (bars[key]) {
                bars[key].fill.style.width = percent;
                bars[key].text.textContent = percent;
            }
        });
    }

    function resetBars() {
        Object.values(bars).forEach(b => {
            b.fill.style.width = '0%';
            b.text.textContent = '0%';
        });
    }
});
