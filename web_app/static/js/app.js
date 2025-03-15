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
// Chart visualization has been removed

// Initialize the application
function init() {
    setupEventListeners();
    
    // Check if we have initial file data from the server
    if (window.initialFileData) {
        console.log('Initializing with file data:', window.initialFileData);
        currentFileId = window.initialFileData.fileId;
        
        // Hide the upload section
        uploadSection.classList.add('hidden');
        
        // Handle different file states
        const status = window.initialFileData.status.status;
        
        if (status === 'analyzing') {
            // Show analysis section with progress
            analysisSection.classList.remove('hidden');
            analysisProgress.classList.remove('hidden');
            analysisResults.classList.add('hidden');
            // Start polling for status updates
            startStatusPolling();
        } 
        else if (status === 'analyzed') {
            // Show analysis results
            analysisSection.classList.remove('hidden');
            analysisProgress.classList.add('hidden');
            analysisResults.classList.remove('hidden');
            displayAnalysisResults(window.initialFileData.status);
        } 
        else if (status === 'generating_video') {
            // Show video generation in progress
            videoSection.classList.remove('hidden');
            videoProgress.classList.remove('hidden');
            videoResults.classList.add('hidden');
            // Start polling for status updates
            startStatusPolling();
        } 
        else if (status === 'completed') {
            // Show completed video
            videoSection.classList.remove('hidden');
            videoProgress.classList.add('hidden');
            videoResults.classList.remove('hidden');
            displayVideo(window.initialFileData.status);
        }
        else if (status === 'error') {
            // Show error message and reset to upload
            alert(`Error: ${window.initialFileData.status.error || 'Unknown error'}`);
            resetToUpload();
        }
    }
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
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        },
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
        method: 'POST',
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
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
    
    // Update URL to reflect we're viewing the file
    if (currentFileId) {
        const newUrl = `/file/${currentFileId}`;
        window.history.pushState({ fileId: currentFileId }, '', newUrl);
        console.log(`URL updated to: ${newUrl}`);
    }
    
    // Poll for status updates every 500ms to catch all progress updates
    statusCheckInterval = setInterval(() => {
        fetch(`/status/${currentFileId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Update progress based on status and progress information
                // Use the status directly (it's already uppercase from the backend)
                switch (data.status) {
                    case 'ANALYZING':
                        // Show real progress from backend if available
                        if (data.progress) {
                            setProgress(analysisProgress, data.progress.percent);
                            // Update progress status text if available
                            updateProgressStatus(analysisProgress, data.progress.status);
                        }
                        break;
                        
                    case 'ANALYZED':
                        // Analysis complete, show results
                        setProgress(analysisProgress, 100);
                        updateProgressStatus(analysisProgress, 'Analysis complete');
                        
                        // Check if we have valid stats in the beat detection task
                        const beatTask = data.beat_detection_task || {};
                        const hasValidStats = beatTask && beatTask.stats && Object.keys(beatTask.stats).length > 0;
                        
                        if (hasValidStats) {
                            // We have valid stats, so we can stop polling and display the results
                            console.log('Valid stats found, displaying analysis results');
                            setTimeout(() => {
                                analysisProgress.classList.add('hidden');
                                analysisResults.classList.remove('hidden');
                                displayAnalysisResults(data);
                            }, 500);
                            clearInterval(statusCheckInterval);
                        } else {
                            // No valid stats yet, continue polling
                            console.warn('Status is ANALYZED but no valid stats found yet. Continuing to poll...');
                            // Don't clear the interval or hide the progress bar yet
                        }
                        break;
                        
                    case 'GENERATING_VIDEO':
                        // Show real progress from backend if available
                        console.log('Received generating_video status update');
                        console.log('Full response data:', data);
                        
                        // We don't need to update the URL since we're already on /file/{fileId}
                        
                        if (data.progress) {
                            console.log(`Video progress update: ${data.progress.status} - ${data.progress.percent}%`);
                            
                            // Log the current progress bar state before updating
                            const progressFill = videoProgress.querySelector('.progress-fill');
                            const currentWidth = progressFill.style.width;
                            console.log(`Current progress bar width: ${currentWidth}`);
                            
                            // Force a reflow before updating the progress bar
                            // This helps ensure the CSS transition is triggered
                            void progressFill.offsetWidth;
                            
                            // Update the progress bar with a small delay to ensure it's rendered
                            setTimeout(() => {
                                // Update the progress bar
                                setProgress(videoProgress, data.progress.percent);
                                
                                // Update progress status text if available
                                updateProgressStatus(videoProgress, data.progress.status);
                                
                                // Log the progress bar state after updating
                                const newWidth = progressFill.style.width;
                                console.log(`New progress bar width: ${newWidth}`);
                            }, 10);
                        } else {
                            console.log('Video progress data not available in response');
                        }
                        break;
                        
                    case 'COMPLETED':
                        // Video generation complete
                        setProgress(videoProgress, 100);
                        updateProgressStatus(videoProgress, 'Video generation complete');
                        // We don't need to update the URL since we're already on /file/{fileId}
                        setTimeout(() => {
                            videoProgress.classList.add('hidden');
                            videoResults.classList.remove('hidden');
                            displayVideo(data);
                        }, 500);
                        clearInterval(statusCheckInterval);
                        break;
                        
                    case 'ERROR':
                        // Error occurred
                        if (data.progress && data.progress.status) {
                            alert(`Error: ${data.progress.status}`);
                        } else {
                            alert(`Error: ${data.error}`);
                        }
                        resetToUpload();
                        clearInterval(statusCheckInterval);
                        break;
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                clearInterval(statusCheckInterval);
            });
    }, 500);  // Poll more frequently (every 500ms) to catch all updates
}

// Display analysis results
function displayAnalysisResults(data) {
    console.log('Displaying analysis results with data:', JSON.stringify(data, null, 2));
    
    // Get beat detection task data if available
    const beatTask = data.beat_detection_task || {};
    console.log('Beat detection task:', JSON.stringify(beatTask, null, 2));
    
    // Only look for stats inside the beat detection task object
    let stats = {};
    if (beatTask && beatTask.stats) {
        stats = beatTask.stats;
        console.log('Found stats in beatTask.stats');
    }
    
    console.log('Final stats object used:', JSON.stringify(stats, null, 2));
    
    // Check if we have valid stats
    if (Object.keys(stats).length === 0) {
        console.warn('No stats found in the beat detection task yet. This might be a race condition.');
        console.log('Will continue polling for status updates until stats are available.');
        // Don't return - we'll just show placeholder values and continue polling
    }
    
    // Update result display
    resultBpm.textContent = stats.bpm ? Math.round(stats.bpm) : '--';
    resultTotalBeats.textContent = stats.total_beats || '--';
    resultDuration.textContent = stats.duration ? formatTime(stats.duration) : '--';
    resultMeter.textContent = stats.detected_meter ? `${stats.detected_meter}/4` : '--';
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
        method: 'POST',
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
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
    
    // Reset URL to base
    window.history.pushState({}, '', '/');
    
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
    
    // Ensure value is a number and between 0-100
    const safeValue = Math.min(Math.max(parseFloat(value) || 0, 0), 100);
    
    // Log the progress update
    console.log(`Setting progress to ${safeValue}%`);
    
    // Apply the width change
    progressFill.style.width = `${safeValue}%`;
}

// Update progress status text
function updateProgressStatus(progressElement, statusText) {
    // Check if status text element exists, if not create it
    let statusElement = progressElement.querySelector('.progress-status');
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.className = 'progress-status text-center mt-2';
        progressElement.appendChild(statusElement);
    }
    statusElement.textContent = statusText;
}

// Format time in seconds to MM:SS format
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// No longer needed - URL is updated directly where needed

// Handle browser back/forward navigation
window.addEventListener('popstate', (event) => {
    // If we have state data and we're going back to the root
    if (window.location.pathname === '/' && currentFileId) {
        resetToUpload();
    }
});

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
