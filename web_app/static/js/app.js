// Beat Detection App - Client-side JavaScript

// DOM Elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const fileInfo = document.getElementById('file-info');
const uploadButton = document.getElementById('upload-button');
const uploadProgress = document.getElementById('upload-progress');
const uploadSection = document.getElementById('upload-section');
const analysisSection = document.getElementById('analysis-section');
const analysisProgress = document.getElementById('analysis-progress');
const analysisResults = document.getElementById('analysis-results');
const confirmButton = document.getElementById('confirm-button');
const cancelButton = document.getElementById('cancel-button');
const videoSection = document.getElementById('video-section');
const videoProgress = document.getElementById('video-progress');
const videoResults = document.getElementById('video-results');
const resultVideo = document.getElementById('result-video');
const downloadButton = document.getElementById('download-button');
const restartButton = document.getElementById('restart-button');

// Result display elements
const resultBpm = document.getElementById('result-bpm');
const resultTotalBeats = document.getElementById('result-total-beats');
const resultDuration = document.getElementById('result-duration');
const resultMeter = document.getElementById('result-meter');

// Global state
let currentFileId = null;
let statusCheckInterval = null;
let beatChart = null;

// Initialize the application
function init() {
    setupEventListeners();
}

// Set up all event listeners
function setupEventListeners() {
    // File selection via button
    fileInput.addEventListener('change', handleFileSelection);
    
    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    // Button event listeners
    uploadButton.addEventListener('click', uploadFile);
    confirmButton.addEventListener('click', confirmAnalysis);
    cancelButton.addEventListener('click', resetToUpload);
    downloadButton.addEventListener('click', downloadVideo);
    restartButton.addEventListener('click', resetToUpload);
}

// Prevent default behaviors for drag and drop
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop area when dragging over
function highlight() {
    dropArea.classList.add('highlight');
}

// Remove highlight when dragging leaves
function unhighlight() {
    dropArea.classList.remove('highlight');
}

// Handle file drop
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelection();
    }
}

// Handle file selection (via input or drop)
function handleFileSelection() {
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        
        // Check file type
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const validExtensions = ['mp3', 'wav', 'flac', 'm4a', 'ogg'];
        
        if (!validExtensions.includes(fileExtension)) {
            alert('Please select a valid audio file (MP3, WAV, FLAC, M4A, OGG)');
            fileInput.value = '';
            return;
        }
        
        // Display file info
        fileName.textContent = file.name;
        fileInfo.classList.remove('hidden');
    }
}

// Upload the selected file
function uploadFile() {
    if (!fileInput.files.length) {
        alert('Please select a file first');
        return;
    }
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    // Show upload progress
    fileInfo.classList.add('hidden');
    uploadProgress.classList.remove('hidden');
    setProgress(uploadProgress, 0);
    
    // Upload file
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Store file ID for future requests
        currentFileId = data.file_id;
        
        // Show 100% progress
        setProgress(uploadProgress, 100);
        
        // Start analysis
        setTimeout(() => {
            uploadSection.classList.add('hidden');
            analysisSection.classList.remove('hidden');
            analyzeAudio();
        }, 500);
    })
    .catch(error => {
        console.error('Error uploading file:', error);
        alert('Error uploading file. Please try again.');
        uploadProgress.classList.add('hidden');
        fileInfo.classList.remove('hidden');
    });
}

// Analyze the uploaded audio
function analyzeAudio() {
    if (!currentFileId) {
        alert('No file uploaded');
        return;
    }
    
    // Show analysis progress
    analysisProgress.classList.remove('hidden');
    analysisResults.classList.add('hidden');
    setProgress(analysisProgress, 10);
    
    // Start analysis
    fetch(`/analyze/${currentFileId}`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Start polling for status
        startStatusPolling();
    })
    .catch(error => {
        console.error('Error starting analysis:', error);
        alert('Error starting analysis. Please try again.');
        resetToUpload();
    });
}

// Start polling for processing status
function startStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    // Clear any previous warning messages
    const warningElements = document.querySelectorAll('.warning-text');
    warningElements.forEach(el => el.remove());
    
    let progress = 10;
    
    statusCheckInterval = setInterval(() => {
        fetch(`/status/${currentFileId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Update progress based on status
                switch (data.status) {
                    case 'analyzing':
                        // Simulate progress while analyzing
                        progress = Math.min(progress + 5, 90);
                        setProgress(analysisProgress, progress);
                        break;
                        
                    case 'analyzed':
                        // Analysis complete, show results
                        setProgress(analysisProgress, 100);
                        setTimeout(() => {
                            analysisProgress.classList.add('hidden');
                            analysisResults.classList.remove('hidden');
                            displayAnalysisResults(data);
                        }, 500);
                        clearInterval(statusCheckInterval);
                        break;
                        
                    case 'generating_video':
                        // Video generation in progress
                        progress = Math.min(progress + 5, 90);
                        setProgress(videoProgress, progress);
                        break;
                        
                    case 'completed':
                        // Video generation complete
                        setProgress(videoProgress, 100);
                        setTimeout(() => {
                            videoProgress.classList.add('hidden');
                            videoResults.classList.remove('hidden');
                            displayVideo(data);
                        }, 500);
                        clearInterval(statusCheckInterval);
                        break;
                        
                    case 'error':
                        // Error occurred
                        alert(`Error: ${data.error}`);
                        resetToUpload();
                        clearInterval(statusCheckInterval);
                        break;
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                clearInterval(statusCheckInterval);
            });
    }, 1000);
}

// Display analysis results
function displayAnalysisResults(data) {
    if (!data.stats) {
        alert('No analysis results available');
        return;
    }
    
    const stats = data.stats;
    
    // Update result display
    resultBpm.textContent = Math.round(stats.bpm);
    resultTotalBeats.textContent = stats.total_beats;
    resultDuration.textContent = formatTime(stats.duration);
    resultMeter.textContent = `${stats.detected_meter}/4`;
    
    // Create a simple chart to visualize beats
    createBeatChart(stats);
}

// Create a chart to visualize beats
function createBeatChart(stats) {
    const ctx = document.getElementById('beat-chart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (beatChart) {
        beatChart.destroy();
    }
    
    // Create dummy beat data for visualization
    const beatCount = stats.total_beats;
    const duration = stats.duration;
    const bpm = stats.bpm;
    
    // Create labels and data points
    const labels = [];
    const data = [];
    
    // Generate beat points (simplified visualization)
    const beatInterval = 60 / bpm; // seconds between beats
    for (let i = 0; i < Math.min(beatCount, 30); i++) {
        const time = i * beatInterval;
        labels.push(formatTime(time));
        
        // Create a pattern where downbeats are higher
        if (i % stats.detected_meter === 0) {
            data.push(100); // Downbeat
        } else {
            data.push(70); // Regular beat
        }
    }
    
    // Create chart
    beatChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Beat Intensity',
                data: data,
                backgroundColor: function(context) {
                    const index = context.dataIndex;
                    return index % stats.detected_meter === 0 ? 
                        'rgba(74, 111, 165, 0.8)' : 'rgba(108, 117, 125, 0.6)';
                },
                borderColor: function(context) {
                    const index = context.dataIndex;
                    return index % stats.detected_meter === 0 ? 
                        'rgba(74, 111, 165, 1)' : 'rgba(108, 117, 125, 0.8)';
                },
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 120,
                    display: false
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Confirm analysis and generate video
function confirmAnalysis() {
    if (!currentFileId) {
        alert('No file uploaded');
        return;
    }
    
    // Show video section
    analysisSection.classList.add('hidden');
    videoSection.classList.remove('hidden');
    videoProgress.classList.remove('hidden');
    videoResults.classList.add('hidden');
    setProgress(videoProgress, 10);
    
    // Start video generation
    fetch(`/confirm/${currentFileId}`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Start polling for status
        startStatusPolling();
    })
    .catch(error => {
        console.error('Error confirming analysis:', error);
        alert('Error confirming analysis. Please try again.');
        resetToUpload();
    });
}

// Display the generated video
function displayVideo(data) {
    if (!data.video_file) {
        alert('No video file available');
        return;
    }
    
    // Check if there was a warning during video generation
    if (data.warning) {
        console.warn('Warning during video generation:', data.warning);
        // Show a warning message to the user
        const warningElement = document.createElement('p');
        warningElement.className = 'warning-text';
        warningElement.textContent = 'Note: The video was generated successfully, but with a minor warning that does not affect playback.';
        videoResults.insertBefore(warningElement, videoResults.firstChild);
    }
    
    // Set video source
    resultVideo.src = `/download/${currentFileId}`;
    resultVideo.load();
}

// Download the generated video
function downloadVideo() {
    if (!currentFileId) {
        alert('No video available');
        return;
    }
    
    // Open download link in new tab
    window.open(`/download/${currentFileId}`, '_blank');
}

// Reset to upload screen
function resetToUpload() {
    // Clear file input
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    uploadProgress.classList.add('hidden');
    
    // Reset sections
    uploadSection.classList.remove('hidden');
    analysisSection.classList.add('hidden');
    videoSection.classList.add('hidden');
    
    // Reset progress bars
    setProgress(uploadProgress, 0);
    setProgress(analysisProgress, 0);
    setProgress(videoProgress, 0);
    
    // Clear status check interval
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    
    // Clear file ID
    currentFileId = null;
}

// Set progress bar value
function setProgress(progressElement, value) {
    const progressFill = progressElement.querySelector('.progress-fill');
    progressFill.style.width = `${value}%`;
}

// Format time in seconds to MM:SS format
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
